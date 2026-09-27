"""
============================================================
GNN-GUIDED SALP ALGORITHMS - batched (GPU or CPU)
============================================================

Four optimizers:

    GNNLXSSA      GNN-guided LX-SSA          (ported from the CPU code)
    GNNLXSSA_UQ   + deep-ensemble trust gate (ported from the CPU code)
    GNNQASSA      GNN-guided QA-SSA          (DERIVED - see note below)
    GNNQASSA_UQ   + deep-ensemble trust gate (DERIVED - see note below)

>>> IMPORTANT - PROVENANCE ------------------------------------------
The uploaded package contains GNNLXSSA and GNNLXSSA_UQ only. There is
no GNN-QA-SSA in it (its ALGORITHM_NAMES list ends at "GNNLXSSA_UQ").
GNNQASSA and GNNQASSA_UQ below are therefore NEW CODE, written here by
analogy: the GNN machinery - surrogate, screening, guidance, trust gate,
repair, replay buffer, fine-tuning - is identical to the LX variants,
and only the follower candidate operator is swapped from the Laplace
perturbation to QA-SSA's quadratic-interpolation vertex. Check that
this matches what you intend before using it in a paper.
---------------------------------------------------------------------

Algorithm 2 of the paper is preserved: single leader move (Eq. 6),
follower chain, guided candidate (Eq. 32) using the trained DIRECTION
head, boundary + spacing repair, surrogate screening with exact
evaluation of the top-mu fraction plus a random exploration sample,
replay buffer and periodic online fine-tuning.

BATCHING NOTES
--------------
* R independent runs advance in lockstep; each run owns its own
  surrogate (weights carry a leading run axis) and its own buffer.
* The UQ gate is a TRUE gate: the ensemble predicts first, a gate mask
  is built, and the exact objective is called ONLY on the gated
  candidates. Because the number of gated candidates differs per run
  and a batch must be rectangular, the gather is padded to the widest
  run with a duplicate of that run's first gated candidate - a
  duplicate costs one extra column, and writing its (identical) value
  twice is harmless. Ungated candidates are never evaluated exactly.

  Two counts are therefore reported, and they mean different things:
      Evaluations   - gated candidates, i.e. exact values the optimizer
                      actually used. This is the algorithmic budget.
      ObjectiveCalls- exact objective computations performed, including
                      the padding columns. This is the compute cost.
  For every non-UQ optimizer the two are equal.
* Initial layouts use batched rejection sampling with a spacing-repair
  fallback (the CPU code adds a constructive ring fallback for large N).
============================================================
"""

import math

import numpy as np
from backend import xp, DTYPE, get_rng, USING_GPU, asnumpy
from algorithms_gpu import _Base, _eval_flat, ALGORITHMS, DEFAULT_KWARGS
from surrogate_gpu import BatchedGNWM, build_graphs, SDTYPE
import objective_gpu as OBJ

# wake-model constants shared with the surrogate's graph features
_WTR, _WTK = OBJ.R, OBJ.K
_ALPHA_CONE = float(np.arctan(_WTK))
_AJ = float(1.0 - np.sqrt(1.0 - OBJ.CT))
_THETAS = xp.asarray(np.deg2rad(np.arange(0, 360, 15) + 7.5), dtype=DTYPE)


def _dataset_globals(dataset):
    """omega, ideal power and direction-averaged scale for the graph features."""
    cfg = OBJ._cfg(dataset)
    omega, psi, ideal = cfg["omega"], cfg["psi"], cfg["ideal"]
    w = float(xp.sum(omega))
    psibar = float(xp.sum(omega * psi) / w) if w > 0 else float(xp.mean(psi))
    return omega, ideal, psibar


# ===============================================================
# constraint repair  (Sec. III-E)
# ===============================================================
def boundary_repair(P, r):
    """P : (...,N,2) in place-safe form."""
    nrm = xp.sqrt(xp.sum(P * P, axis=-1))
    scale = xp.where(nrm > r, r / xp.maximum(nrm, 1e-12), 1.0)
    return P * scale[..., None]


def spacing_repair(P, r, passes=1, min_dist=None):
    """Faithful port: sequential i<j sweep, each pair pushed apart in place.

    The pair loop is kept sequential (later pairs see earlier pushes,
    exactly as in the CPU code); it is vectorised over every leading
    axis, so the loop length is N(N-1)/2 regardless of batch size.
    """
    md = 8.0 * _WTR if min_dist is None else min_dist
    n = P.shape[-2]
    for _ in range(passes):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                v = P[..., i, :] - P[..., j, :]
                d = xp.sqrt(xp.sum(v * v, axis=-1))
                close = d < md
                if not USING_GPU and not bool(xp.any(close)):
                    continue        # cheap skip on CPU; on GPU the sync that
                moved = True        # this test needs costs more than the work
                u = xp.where((d > 1e-9)[..., None],
                             v / xp.maximum(d, 1e-12)[..., None],
                             xp.asarray([1.0, 0.0], dtype=P.dtype))
                shift = (0.5 * (md - d) + 1e-6)[..., None] * close[..., None]
                P[..., i, :] = P[..., i, :] + u * shift
                P[..., j, :] = P[..., j, :] - u * shift
        if not moved:
            break
        P = boundary_repair(P, r)
    return P


def feasible_layouts(rng, shape, n, r, tries=30, min_dist=None):
    """Batched rejection sampling; leftovers fixed by spacing repair.

    shape : leading dimensions, e.g. (R, P). Returns (..., n, 2).
    """
    md = 8.0 * _WTR if min_dist is None else min_dist
    P = xp.zeros(shape + (n, 2), dtype=DTYPE)

    def sample():
        rad = r * xp.sqrt(xp.asarray(rng.random(shape), dtype=DTYPE))
        ang = 2 * math.pi * xp.asarray(rng.random(shape), dtype=DTYPE)
        return xp.stack([rad * xp.cos(ang), rad * xp.sin(ang)], axis=-1)

    for k in range(n):
        cand = sample()
        accepted = xp.zeros(shape, dtype=bool)
        for _ in range(tries):
            if k == 0:
                ok = xp.ones(shape, dtype=bool)
            else:
                d = xp.sqrt(xp.sum((P[..., :k, :] - cand[..., None, :]) ** 2,
                                   axis=-1))
                ok = xp.all(d >= md, axis=-1)
            take = ok & (~accepted)
            P[..., k, :] = xp.where(take[..., None], cand, P[..., k, :])
            accepted = accepted | ok
            if not USING_GPU and bool(xp.all(accepted)):
                break               # same reasoning as in spacing_repair
            cand = sample()
        P[..., k, :] = xp.where(accepted[..., None], P[..., k, :], cand)

    P = boundary_repair(P, r)
    return spacing_repair(P, r, passes=8, min_dist=md)


# ===============================================================
# finite-difference improvement directions (Sec. III-D3)
# ===============================================================
def fd_directions(f, X, n, ne, h=5.0):
    """X : (R,S,2n) -> (R,S,n,2) unit descent directions of the true objective.

    Costs 4n exact evaluations per sample, charged to the counter exactly
    as the CPU code charges them.
    """
    R, S, dim = X.shape
    eye = xp.eye(dim, dtype=DTYPE) * h
    plus = X[:, :, None, :] + eye[None, None]
    minus = X[:, :, None, :] - eye[None, None]
    both = xp.concatenate([plus, minus], axis=2).reshape(R * S * 2 * dim, dim)
    vals = _eval_flat(f, both, ne, per_run=0).reshape(R, S, 2, dim)
    ne[0] += 2 * dim * S            # 4n evals per sample, S samples
    g = (vals[:, :, 0] - vals[:, :, 1]) / (2 * h)
    g = -g.reshape(R, S, n, 2)
    nrm = xp.sqrt(xp.sum(g * g, axis=-1, keepdims=True))
    return g / xp.maximum(nrm, 1e-12)


# ===============================================================
# base class
# ===============================================================
class _GNNSalpBase(_Base):
    """Algorithm 2 with a pluggable candidate operator and trust rule."""

    uq = False
    operator = "laplace"          # or "quadratic"

    def optimize(self, f, dim, lb, ub, n_runs, pop, iters, seed):
        rng = get_rng(seed)
        n = dim // 2
        r = ub

        phi = self.kw.get("phi", 0.0)
        chi = self.kw.get("chi", 1.0)
        mu = self.kw.get("mu", 0.2)
        tau0 = self.kw.get("tau0", 0.05)
        hidden = self.kw.get("hidden", 64)
        layers = self.kw.get("mp_layers", 3)
        n_pretrain = self.kw.get("n_pretrain", 40)
        finetune_every = self.kw.get("finetune_every", 5)
        lambda_g = self.kw.get("lambda_g", 0.5)
        fd_fraction = self.kw.get("fd_fraction", 0.25)
        fd_max = self.kw.get("fd_max", 16)
        dataset = self.kw.get("dataset", 1)
        # Hard cap on exact evaluations, used by the fixed-budget regime. The
        # UQ variants cannot have their iteration count derived from a budget
        # in advance (the gate is data-dependent), so they are run long and
        # stopped as soon as the slowest-spending run reaches the cap.
        max_evals = self.kw.get("max_evals", None)
        buffer_cap = self.kw.get("buffer_cap", 256)

        n_models = self.kw.get("n_models", 5) if self.uq else 1
        sigma_thr = self.kw.get("sigma_threshold", 0.5)
        retrain_interval = self.kw.get("retrain_interval", 20)
        retrain_epochs = self.kw.get("retrain_epochs", 2)

        omega, ideal, psibar = _dataset_globals(dataset)
        norm = 100.0 / (ideal * n)
        feas_cap = ideal * n
        min_dist = OBJ.min_spacing(dataset)

        gnn = BatchedGNWM(n_runs, hidden=hidden, layers=layers,
                          seed=seed, n_models=n_models)

        ne = [0]                       # exact evaluations shared by every run
        gated = xp.zeros(n_runs, dtype=DTYPE)   # gate-admitted, per run
        calls = xp.zeros(n_runs, dtype=DTYPE)   # exact objective calls performed
        self.n_surrogate_evals = 0            # inference (screening/guidance)
        self.n_surrogate_train = 0            # training forward passes
        self.n_finetunes = 0
        # Calibration evidence for the UQ papers. Every gated candidate gives
        # a matched pair: what the ensemble predicted and how far off it was.
        # Kept as (sigma, |error|, predicted, exact) rows in real wake-loss
        # units, padding columns excluded. This is what turns "the gate is
        # cheaper" into "the uncertainty is informative" - without it, a
        # reviewer has no way to tell whether the ensemble spread tracks the
        # surrogate's actual error or is decorative.
        self.uq_calibration = []
        self.gate_admitted = []       # gate admissions per iteration, per run

        def graphs(P):
            """P : (R,G,n,2) -> surrogate inputs, replicated per ensemble member."""
            Xn, Ea, mk = build_graphs(P, r, omega, _THETAS, _ALPHA_CONE, _AJ,
                                      _WTK, _WTR, psibar)
            if n_models == 1:
                return Xn, Ea, mk
            rep = lambda A: xp.concatenate([A] * n_models, axis=0)
            return rep(Xn), rep(Ea), rep(mk)

        def surro(P, want_dir=False):
            """mean prediction (R,G) [+ std (R,G)] [+ guidance (R,G,n,2)]."""
            Xn, Ea, mk = graphs(P)
            if want_dir:
                pred, gout = gnn.guidance(Xn, Ea, mk)
            else:
                pred, gout = gnn.predict(Xn, Ea, mk), None
            self.n_surrogate_evals += int(P.shape[1]) * n_models
            if n_models == 1:
                return (pred, xp.zeros_like(pred),
                        gout if gout is None else gout)
            pred = pred.reshape(n_models, n_runs, -1)
            mean, std = xp.mean(pred, axis=0), xp.std(pred, axis=0)
            if gout is None:
                return mean, std, None
            g = xp.mean(gout.reshape((n_models, n_runs) + gout.shape[1:]), axis=0)
            nrm = xp.sqrt(xp.sum(g * g, axis=-1, keepdims=True))
            return mean, std, g / xp.maximum(nrm, 1e-12)

        # ---------------- initial population ----------------
        P0 = feasible_layouts(rng, (n_runs, pop), n, r, min_dist=min_dist)
        popX = P0.reshape(n_runs, pop, dim)
        fit = _eval_flat(f, popX.reshape(n_runs * pop, dim), ne,
                         per_run=pop).reshape(n_runs, pop)

        order = xp.argsort(fit, axis=1)
        popX = xp.take_along_axis(popX, order[..., None], axis=1)
        fit = xp.take_along_axis(fit, order, axis=1)
        best_x, best_f = popX[:, 0, :].copy(), fit[:, 0].copy()
        H = best_x.copy()
        curves = [best_f.copy()]
        self.eval_axis = []
        self._track(ne, n_runs, gated if self.uq else None)

        # buffers of raw positions + targets (graphs rebuilt on demand)
        buf_P = [P0]
        buf_y = [fit * norm]
        buf_g = [xp.full((n_runs, pop, n, 2), math.nan, dtype=SDTYPE)]

        # ---------------- pre-training ----------------
        n_fd = min(fd_max, max(1, int(fd_fraction * (n_pretrain + pop))))
        pre = feasible_layouts(rng, (n_runs, n_pretrain), n, r, min_dist=min_dist)
        preX = pre.reshape(n_runs, n_pretrain, dim)
        pref = _eval_flat(f, preX.reshape(-1, dim), ne,
                          per_run=n_pretrain).reshape(n_runs, n_pretrain)

        gfd = xp.full((n_runs, n_pretrain, n, 2), math.nan, dtype=SDTYPE)
        if n_fd > 0:
            k = min(n_fd, n_pretrain)
            g = fd_directions(f, preX[:, :k], n, ne)
            ok = (pref[:, :k] < feas_cap)[..., None, None]
            gfd[:, :k] = xp.where(ok, g.astype(SDTYPE), math.nan)

        buf_P.append(pre)
        buf_y.append(pref * norm)
        buf_g.append(gfd)

        gv = xp.min(pref, axis=1)
        imp = gv < best_f
        best_x = xp.where(imp[:, None],
                          xp.take_along_axis(preX, xp.argmin(pref, axis=1)[:, None, None],
                                             axis=1)[:, 0, :], best_x)
        best_f = xp.where(imp, gv, best_f)
        H = best_x.copy()

        def train(Psub, ysub, gsub, epochs, batch=16):
            """Train only on labelled, penalty-free samples - the CPU code
            achieves the same by never buffering the others."""
            valid = xp.isfinite(ysub) & (ysub < feas_cap * norm)
            if not bool(xp.any(valid)):
                return
            ysub = xp.where(valid, ysub, 0.0)
            Xn, Ea, mk = graphs(Psub)
            rep = lambda A: A if n_models == 1 else xp.concatenate([A] * n_models, 0)
            gg = None if gsub is None else rep(gsub)
            gnn.train(Xn, Ea, mk, rep(ysub).astype(SDTYPE), gfd=gg,
                      epochs=epochs, batch=batch, lambda_g=lambda_g, rng=rng,
                      valid=rep(valid))

        train(xp.concatenate(buf_P, axis=1),
              xp.concatenate(buf_y, axis=1),
              xp.concatenate(buf_g, axis=1), epochs=25)

        n_exact = max(1, int(np.ceil(mu * 2 * pop)))
        since_retrain = 0

        # ---------------- main loop ----------------
        for t in range(1, iters + 1):
            r1 = 2.0 * float(np.exp(-((4.0 * t / iters) ** 2)))
            tau = tau0 * (1.0 - t / iters)

            # ---- leader (Eq. 6) + follower chain ----
            moved = popX.copy()
            r2 = xp.asarray(rng.random((n_runs, dim)), dtype=DTYPE)
            r3 = xp.asarray(rng.random((n_runs, dim)), dtype=DTYPE)
            step = r1 * ((ub - lb) * r2 + lb)
            moved[:, 0, :] = xp.where(r3 >= 0.5, H + step, H - step)
            for i in range(1, pop):
                moved[:, i, :] = 0.5 * (popX[:, i, :] + moved[:, i - 1, :])

            # ---- guided candidates (Eq. 32 / QA analogue) ----
            _, _, ghat = surro(moved.reshape(n_runs, pop, n, 2), want_dir=True)
            ghat = ghat.reshape(n_runs, pop, dim).astype(DTYPE)

            if self.operator == "laplace":
                z = xp.asarray(rng.random((n_runs, pop, 1)), dtype=DTYPE)
                gamma = xp.where(z <= 0.5,
                                 phi - chi * xp.log(xp.maximum(z, 1e-12)),
                                 phi + chi * xp.log(z))
                base = moved + gamma * (H[:, None, :] - moved)
            else:
                base = self._quadratic_base(rng, moved, popX, fit, best_f,
                                            H, n_runs, pop, dim)

            cands = base + (tau * r) * ghat

            # ---- repair both sets ----
            mv = spacing_repair(boundary_repair(moved.reshape(n_runs, pop, n, 2), r),
                                r, passes=1, min_dist=min_dist)
            cd = spacing_repair(boundary_repair(cands.reshape(n_runs, pop, n, 2), r),
                                r, passes=1, min_dist=min_dist)
            moved = xp.clip(mv.reshape(n_runs, pop, dim), lb, ub)
            cands = xp.clip(cd.reshape(n_runs, pop, dim), lb, ub)

            allc = xp.concatenate([moved, cands], axis=1)          # (R,2P,dim)
            allP = allc.reshape(n_runs, 2 * pop, n, 2)

            mean, std, _ = surro(allP)
            score = mean.astype(DTYPE) / norm

            if not self.uq:
                # ---- top-mu screening + 2 random probes ----
                idx = xp.argsort(mean, axis=1)[:, :n_exact]
                extra = xp.asarray(rng.integers(0, 2 * pop, (n_runs, 2)))
                pick = xp.concatenate([idx, extra], axis=1)         # (R,k)
                sel = xp.take_along_axis(allc, pick[..., None], axis=1)
                k = pick.shape[1]
                vals = _eval_flat(f, sel.reshape(n_runs * k, dim), ne,
                                  per_run=k).reshape(n_runs, k)
                calls = calls + k
                score = _scatter(score, pick, vals)
                newP = sel.reshape(n_runs, k, n, 2)
                newy = vals * norm
                # the incumbent may ONLY come from exact evaluations
                exact_vals, exact_x = vals, sel
            else:
                # ---- TRUE uncertainty gate: evaluate only what it admits ----
                gate = std > sigma_thr
                gate = _set_at(gate, xp.argmin(mean, axis=1), True)  # verify best
                n_gated = xp.sum(gate, axis=1).astype(DTYPE)

                # HARD BUDGET CAP. Under the fixed-budget regime a run may not
                # be able to afford every candidate the gate would admit. The
                # admissions are then truncated to whatever budget is left,
                # spending it on the most promising candidates rather than
                # overshooting the cap. A run with nothing left admits nothing
                # and cannot improve further - `best_f` only ever takes values
                # from exact evaluations - so it holds its incumbent while the
                # others finish, which is exactly what "best found within
                # budget B" means.
                if max_evals is not None:
                    spent = ne[0] + gated
                    n_gated = xp.minimum(n_gated,
                                         xp.maximum(max_evals - spent, 0.0))

                width = int(xp.max(n_gated))
                if width == 0:                  # every run has exhausted B
                    curves.append(best_f.copy())
                    self._track(ne, n_runs, gated)
                    break

                # Gated candidates ordered by ASCENDING PREDICTED MEAN, so a
                # truncated run keeps the ones the surrogate rates best and the
                # forced generation-best (the argmin of mean) always comes
                # first. Padding columns repeat that first candidate, so the
                # duplicate carries the same exact value and no ungated
                # candidate is ever evaluated.
                col = xp.arange(2 * pop)
                rank = xp.argsort(xp.argsort(mean, axis=1), axis=1)
                key = xp.where(gate, rank, 2 * pop + rank)
                order = xp.argsort(key, axis=1)[:, :width]
                keep = col[:width][None, :] < n_gated[:, None]
                idx = xp.where(keep, order, order[:, :1])

                sel = xp.take_along_axis(allc, idx[..., None], axis=1)
                vals_k = _eval_flat(f, sel.reshape(n_runs * width, dim), ne,
                                    per_run=0).reshape(n_runs, width)

                gated = gated + n_gated              # budget: what the gate used
                calls = calls + width                # compute: what was evaluated

                score = _scatter(score, idx, vals_k)

                # Matched prediction/outcome pairs, real units, real entries
                # only (`keep` drops the padding duplicates so they cannot
                # weight the calibration twice).
                mu_k = xp.take_along_axis(mean, idx, axis=1) / norm
                sg_k = xp.take_along_axis(std, idx, axis=1) / norm
                m = keep.reshape(-1)
                # The 5th column flags whether the exact value is a FEASIBLE
                # layout. Infeasible ones carry the 1e10 constraint penalty, so
                # their objective is ~1e23 and the surrogate - which is trained
                # only on feasible layouts and predicts a percentage of ideal
                # power - cannot possibly track them. Including those pairs
                # would swamp every calibration statistic with numbers that say
                # nothing about the surrogate. They are kept rather than
                # dropped so the analysis can report how many there were.
                self.uq_calibration.append(xp.stack([
                    sg_k.reshape(-1)[m],
                    xp.abs(mu_k - vals_k).reshape(-1)[m],
                    mu_k.reshape(-1)[m],
                    vals_k.reshape(-1)[m],
                    (vals_k < feas_cap).astype(DTYPE).reshape(-1)[m]],
                    axis=1))
                self.gate_admitted.append(n_gated / float(2 * pop))
                newP = sel.reshape(n_runs, width, n, 2)
                newy = vals_k * norm
                exact_vals, exact_x = vals_k, sel

            # ---- incumbent: exact values only, so best_f and the curve
            #      are always ground truth (as in the CPU implementation) ----
            gi = xp.argmin(exact_vals, axis=1)
            gv = xp.min(exact_vals, axis=1)
            imp = gv < best_f
            best_x = xp.where(imp[:, None],
                              xp.take_along_axis(exact_x, gi[:, None, None],
                                                 axis=1)[:, 0, :], best_x)
            best_f = xp.where(imp, gv, best_f)
            H = best_x.copy()

            # ---- replay buffer ----
            buf_P.append(newP)
            buf_y.append(newy)
            buf_g.append(xp.full(newP.shape[:2] + (n, 2), math.nan, dtype=SDTYPE))
            total = sum(b.shape[1] for b in buf_P)
            while total > buffer_cap and len(buf_P) > 1:
                total -= buf_P[0].shape[1]
                buf_P.pop(0); buf_y.pop(0); buf_g.pop(0)

            # ---- greedy selection per salp ----
            a = score[:, :pop]
            b = score[:, pop:]
            take = (a <= b)[..., None]
            popX = xp.where(take, moved, cands)
            fit = xp.where(take[..., 0], a, b)

            # ---- periodic fine-tuning ----
            # LX/QA screening variants: every `finetune_every` iterations.
            # UQ variants: after `retrain_interval` new exact samples, which is
            # how the CPU implementation triggers it.
            if self.uq:
                since_retrain += float(xp.mean(n_gated))
                due = since_retrain >= retrain_interval
            else:
                due = (t % finetune_every == 0)
            if due:
                Pall = xp.concatenate(buf_P, axis=1)
                yall = xp.concatenate(buf_y, axis=1)
                gall = xp.concatenate(buf_g, axis=1)
                S = Pall.shape[1]
                # argsort of random keys instead of rng.permutation:
                # available on every backend, and stays on the device
                sub = xp.argsort(xp.asarray(rng.random(S)))[:min(S, 96)]
                train(Pall[:, sub], yall[:, sub], gall[:, sub],
                      epochs=(retrain_epochs if self.uq else 2))
                self.n_finetunes += 1
                since_retrain = 0

            curves.append(best_f.copy())
            self._track(ne, n_runs, gated if self.uq else None)

            if max_evals is not None:
                # Non-UQ GNN under a cap (not used by the runner, which derives
                # an exact iteration count for it) and a belt-and-braces guard
                # for the UQ path, whose truncation above already holds it at
                # the cap.
                spent = (ne[0] + gated) if self.uq else ne[0]
                if float(xp.max(xp.asarray(spent))) >= max_evals:
                    break

        self.n_surrogate_train = int(getattr(gnn, "n_train_forwards", 0))
        if self.uq and self.uq_calibration:
            self.uq_calibration = asnumpy(
                xp.concatenate(self.uq_calibration, axis=0))
            self.gate_rate = asnumpy(
                xp.mean(xp.stack(self.gate_admitted, axis=0), axis=0))
        else:
            self.uq_calibration = np.zeros((0, 5))
            self.gate_rate = np.zeros(n_runs)

        # `Evaluations`: exact values the optimizer used (the budget).
        # `n_objective_calls`: exact objective computations performed.
        # They differ only for the UQ variants, and only by the padding.
        total = (ne[0] + gated) if self.uq else ne[0]
        self.n_exact_evals = total
        self.n_objective_calls = (ne[0] + calls) if self.uq else total
        return best_x, best_f, xp.stack(curves, axis=1), total

    # ---- QA-SSA vertex, used by the derived GNN-QA variants ----
    @staticmethod
    def _quadratic_base(rng, moved, popX, fit, best_f, H, n_runs, pop, dim):
        ar = xp.arange(n_runs)[:, None]
        j1 = xp.asarray(rng.integers(1, pop, (n_runs, pop)))
        j2 = xp.asarray(rng.integers(1, max(pop - 1, 2), (n_runs, pop)))
        j2 = xp.where(j2 >= j1, j2 + 1, j2)
        j2 = xp.clip(j2, 1, pop - 1)

        Bp, Cp = popX[ar, j1], popX[ar, j2]                 # (R,P,dim)
        fB, fC = fit[ar, j1][..., None], fit[ar, j2][..., None]
        fH = best_f[:, None, None]
        Hh = H[:, None, :]

        num = ((Bp ** 2 - Cp ** 2) * fH
               + (Cp ** 2 - Hh ** 2) * fB
               + (Hh ** 2 - Bp ** 2) * fC)
        den = ((Bp - Cp) * fH + (Cp - Hh) * fB + (Hh - Bp) * fC)
        scale = xp.maximum(xp.abs(fH), 1.0)
        safe = xp.abs(den) > (1e-12 * scale)
        return xp.where(safe, 0.5 * num / xp.where(safe, den, 1.0), Hh)


def _scatter(base, idx, vals):
    """base : (R,M) ; idx, vals : (R,k) -> base with vals written at idx."""
    out = base.copy()
    r = xp.arange(base.shape[0])[:, None]
    out[r, idx] = vals
    return out


def _set_at(mask, idx, value):
    out = mask.copy()
    out[xp.arange(mask.shape[0]), idx] = value
    return out


# ===============================================================
class GNNLXSSA(_GNNSalpBase):
    name = "GNNLXSSA"
    operator = "laplace"
    uq = False


class GNNQASSA(_GNNSalpBase):
    """DERIVED (not in the uploaded package) - see module header."""
    name = "GNNQASSA"
    operator = "quadratic"
    uq = False


class GNNLXSSA_UQ(_GNNSalpBase):
    name = "GNNLXSSA_UQ"
    operator = "laplace"
    uq = True


class GNNQASSA_UQ(_GNNSalpBase):
    """DERIVED (not in the uploaded package) - see module header."""
    name = "GNNQASSA_UQ"
    operator = "quadratic"
    uq = True


_GNN_DEFAULTS = dict(phi=0.0, chi=1.0, mu=0.2, tau0=0.05, hidden=64,
                     mp_layers=3, n_pretrain=40, finetune_every=5,
                     lambda_g=0.5, fd_fraction=0.25, fd_max=16, dataset=1)
_UQ_DEFAULTS = dict(_GNN_DEFAULTS, n_models=5, sigma_threshold=0.5,
                    retrain_interval=20, retrain_epochs=2)

for _cls, _kw in ((GNNLXSSA, _GNN_DEFAULTS), (GNNQASSA, _GNN_DEFAULTS),
                  (GNNLXSSA_UQ, _UQ_DEFAULTS), (GNNQASSA_UQ, _UQ_DEFAULTS)):
    ALGORITHMS[_cls.name] = _cls
    DEFAULT_KWARGS[_cls.name] = dict(_kw)
