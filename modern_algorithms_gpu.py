"""
============================================================
MODERN BASELINES: L-SHADE and CMA-ES  (GPU or CPU)
============================================================

Two state-of-the-art continuous optimizers added as baselines, in the same
batched form as algorithms_gpu: `n_runs` independent runs advance in lockstep
and every generation is ONE batched objective call.

Both handle the constraints exactly as the other penalty-based baselines do:
through the penalized objective J = L + P of the campaign, with no repair.

Budget. Both spend exactly the budget of the fixed-budget baselines,
    B = pop * (iters + 1) = 3,030 exact evaluations per run
at the campaign settings (pop = 30, iters = 100). The last generation is
truncated so the count is never exceeded. Their curves are reported at the
same evaluation milestones as the other baselines (pop, 2*pop, ..., B), so
curves, checkpoints and matched-budget readings are directly comparable.

L-SHADE  -  Tanabe & Fukunaga (2014), IEEE CEC, pp. 1658-1665.
    current-to-pbest/1/bin, external archive, success-history memories of
    CR and F (H = 6, weighted Lehmer means, terminal value for CR), and
    linear population size reduction (LPSR) to N_min = 4 over the budget.
    Published values: p = 0.11, H = 6, r_arc = 2.6, N_min = 4, memories
    initialised to 0.5. ONE deliberate deviation: the initial population is
    N_init = pop = 30 (the campaign's population size) instead of 18 * D,
    because 18 * D (up to 648) would leave only ~8 generations within
    3,030 evaluations at the larger farms.

CMA-ES  -  Hansen & Ostermeier (2001); parameter settings of Hansen (2016),
    "The CMA Evolution Strategy: A Tutorial", arXiv:1604.00772, Table 1.
    (mu/mu_w, lambda) with rank-one and rank-mu updates, cumulative step-size
    adaptation, default lambda = 4 + floor(3 ln D), mu = floor(lambda / 2),
    no restarts. Initial mean drawn uniformly in the box, sigma0 =
    (ub - lb) / 4. Candidates are not clipped to the box: anything outside
    the farm disc is penalized by the objective, as for every baseline.
============================================================
"""

import math

import numpy as np

from backend import xp, DTYPE, asnumpy
from algorithms_gpu import _Base, _take_rows


def _milestone_curves(gen_evals, gen_best, pop, iters, n_runs):
    """Resample per-generation best-so-far onto the baselines' milestones.

    gen_evals : list of cumulative evaluations after each generation
    gen_best  : list of (n_runs,) best-so-far values after each generation
    Returns curves (n_runs, iters+1) and the milestone axis.
    """
    E = np.asarray(gen_evals, dtype=float)
    Bv = np.stack([np.asarray(asnumpy(b), dtype=float) for b in gen_best], axis=1)
    miles = pop * (np.arange(iters + 1) + 1.0)
    idx = np.searchsorted(E, miles, side="right") - 1
    idx = np.clip(idx, 0, len(E) - 1)
    return xp.asarray(Bv[:, idx], dtype=DTYPE), miles


class LSHADE(_Base):
    name = "LSHADE"

    def optimize(self, f, dim, lb, ub, n_runs, pop, iters, seed):
        from backend import get_rng
        rng = get_rng(seed)
        p_best = self.kw.get("p", 0.11)
        H = int(self.kw.get("H", 6))
        r_arc = self.kw.get("r_arc", 2.6)
        n_min = int(self.kw.get("n_min", 4))
        n_init = int(self.kw.get("n_init", pop))
        budget = pop * (iters + 1)
        R, D = n_runs, dim
        ar = xp.arange(R)

        X = xp.asarray(rng.uniform(lb, ub, (R, n_init, D)), dtype=DTYPE)
        fit = xp.asarray(f(X.reshape(R * n_init, D)), dtype=DTYPE).reshape(R, n_init)
        nfe = n_init
        NP = n_init

        a_cap = int(round(r_arc * n_init))
        A = xp.zeros((R, max(a_cap, 1), D), dtype=DTYPE)
        a_cnt = np.zeros(R, dtype=int)

        MCR = np.full((R, H), 0.5)
        MF = np.full((R, H), 0.5)
        terminal = np.zeros((R, H), dtype=bool)
        k_mem = np.zeros(R, dtype=int)

        best_f = xp.min(fit, axis=1)
        best_x = _take_rows(X, xp.argmin(fit, axis=1))
        gen_evals, gen_best = [nfe], [best_f.copy()]

        while nfe < budget:
            n_eval = min(NP, budget - nfe)          # truncate the final generation
            # ---- parameters from the success-history memories ----
            ri = rng.integers(0, H, (R, NP))
            mcr = np.take_along_axis(MCR, ri, axis=1)
            term = np.take_along_axis(terminal, ri, axis=1)
            CR = np.clip(rng.normal(mcr, 0.1), 0.0, 1.0)
            CR[term] = 0.0
            mf = np.take_along_axis(MF, ri, axis=1)
            F = mf + 0.1 * np.tan(np.pi * (rng.random((R, NP)) - 0.5))
            bad = F <= 0
            while bad.any():                        # regenerate non-positive F
                F[bad] = mf[bad] + 0.1 * np.tan(np.pi * (rng.random(bad.sum()) - 0.5))
                bad = F <= 0
            F = np.minimum(F, 1.0)

            # ---- current-to-pbest/1 with archive ----
            order = xp.argsort(fit, axis=1)
            n_p = max(2, int(round(p_best * NP)))
            pb = xp.take_along_axis(order, xp.asarray(rng.integers(0, n_p, (R, NP))), axis=1)
            i0 = np.broadcast_to(np.arange(NP), (R, NP))
            r1 = rng.integers(0, NP - 1, (R, NP)); r1 = r1 + (r1 >= i0)
            # r2 from P u A, distinct from i and r1 (rejection, vectorized)
            tot = NP + a_cnt[:, None]
            r2 = (rng.random((R, NP)) * tot).astype(int)
            clash = (r2 == i0) | (r2 == r1)
            while clash.any():
                r2[clash] = (rng.random(clash.sum()) * np.broadcast_to(tot, (R, NP))[clash]).astype(int)
                clash = (r2 == i0) | (r2 == r1)
            PA = xp.concatenate([X, A], axis=1)
            xi = X
            xpb = X[ar[:, None], pb]
            xr1 = X[ar[:, None], xp.asarray(r1)]
            xr2 = PA[ar[:, None], xp.asarray(r2)]
            Fx = xp.asarray(F, dtype=DTYPE)[..., None]
            V = xi + Fx * (xpb - xi) + Fx * (xr1 - xr2)
            # bound handling of the original: midpoint between bound and parent
            V = xp.where(V < lb, (lb + xi) / 2, V)
            V = xp.where(V > ub, (ub + xi) / 2, V)
            # binomial crossover
            jr = xp.asarray(rng.integers(0, D, (R, NP)))[..., None]
            cross = (xp.asarray(rng.random((R, NP, D))) < xp.asarray(CR, dtype=DTYPE)[..., None]) | \
                    (xp.arange(D)[None, None, :] == jr)
            U = xp.where(cross, V, xi)

            # ---- evaluation (first n_eval trial vectors only on the last generation) ----
            ufit = xp.full((R, NP), xp.inf, dtype=DTYPE)
            ufit[:, :n_eval] = xp.asarray(f(U[:, :n_eval].reshape(R * n_eval, D)),
                                          dtype=DTYPE).reshape(R, n_eval)
            nfe += n_eval

            # ---- selection, archive, success memories ----
            better = asnumpy(ufit < fit)
            keep = asnumpy(ufit <= fit)
            dfit = np.abs(asnumpy(fit - ufit))
            Xn = asnumpy(X); Un = asnumpy(U); An = asnumpy(A)
            for r in range(R):
                idx = np.nonzero(better[r])[0]
                for j in idx:                          # archive the replaced parents
                    if a_cap == 0:
                        break
                    if a_cnt[r] < a_cap:
                        An[r, a_cnt[r]] = Xn[r, j]; a_cnt[r] += 1
                    else:
                        An[r, rng.integers(0, a_cap)] = Xn[r, j]
                if len(idx):
                    w = dfit[r, idx]; w = w / w.sum() if w.sum() > 0 else np.full(len(idx), 1 / len(idx))
                    scr, sf = CR[r, idx], F[r, idx]
                    k = k_mem[r]
                    if terminal[r, k] or scr.max() == 0:
                        terminal[r, k] = True
                    else:
                        MCR[r, k] = np.sum(w * scr ** 2) / np.sum(w * scr)
                    MF[r, k] = np.sum(w * sf ** 2) / np.sum(w * sf)
                    k_mem[r] = (k + 1) % H
            A = xp.asarray(An, dtype=DTYPE)
            X = xp.where(xp.asarray(keep)[..., None], U, X)
            fit = xp.where(xp.asarray(keep), ufit, fit)

            gv = xp.min(fit, axis=1)
            imp = gv < best_f
            best_x = xp.where(imp[:, None], _take_rows(X, xp.argmin(fit, axis=1)), best_x)
            best_f = xp.where(imp, gv, best_f)
            gen_evals.append(nfe); gen_best.append(best_f.copy())

            # ---- linear population size reduction ----
            np_next = int(round((n_min - n_init) / budget * nfe + n_init))
            np_next = max(n_min, np_next)
            if np_next < NP:
                srt = xp.argsort(fit, axis=1)[:, :np_next]
                X = xp.take_along_axis(X, srt[..., None], axis=1)
                fit = xp.take_along_axis(fit, srt, axis=1)
                NP = np_next
                new_cap = int(round(r_arc * NP))
                if new_cap < a_cap:                    # shrink archive: drop random members
                    An = asnumpy(A)
                    for r in range(R):
                        if a_cnt[r] > new_cap:
                            keep_i = rng.permutation(a_cnt[r])[:new_cap]
                            An[r, :new_cap] = An[r, keep_i].copy()
                            a_cnt[r] = new_cap
                    A = xp.asarray(An[:, :max(new_cap, 1)], dtype=DTYPE)
                    a_cap = new_cap

        curves, miles = _milestone_curves(gen_evals, gen_best, pop, iters, R)
        self.eval_axis = [np.full(R, m) for m in miles]
        return best_x, best_f, curves, nfe


class CMAES(_Base):
    name = "CMAES"

    def optimize(self, f, dim, lb, ub, n_runs, pop, iters, seed):
        from backend import get_rng
        rng = get_rng(seed)
        R, n = n_runs, dim
        budget = pop * (iters + 1)
        lam = int(self.kw.get("lam", 4 + int(math.floor(3 * math.log(n)))))
        mu = lam // 2
        wts = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        wts = wts / wts.sum()
        mueff = 1.0 / np.sum(wts ** 2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        W = xp.asarray(wts, dtype=DTYPE)

        m = xp.asarray(rng.uniform(lb, ub, (R, n)), dtype=DTYPE)
        sigma = xp.full((R,), self.kw.get("sigma0_frac", 0.25) * (ub - lb), dtype=DTYPE)
        C = xp.broadcast_to(xp.eye(n, dtype=DTYPE), (R, n, n)).copy()
        pc = xp.zeros((R, n), dtype=DTYPE); ps = xp.zeros((R, n), dtype=DTYPE)
        nfe, g = 0, 0
        best_f = xp.full((R,), xp.inf, dtype=DTYPE); best_x = m.copy()
        gen_evals, gen_best = [], []

        while nfe < budget:
            k = min(lam, budget - nfe)
            C = (C + xp.swapaxes(C, 1, 2)) / 2
            evals, B = xp.linalg.eigh(C)
            Dg = xp.sqrt(xp.maximum(evals, 1e-30))                  # (R,n)
            z = xp.asarray(rng.standard_normal((R, lam, n)), dtype=DTYPE)
            y = xp.einsum("rij,rlj->rli", B, z * Dg[:, None, :])     # B diag(D) z
            Xc = m[:, None, :] + sigma[:, None, None] * y
            fx = xp.full((R, lam), xp.inf, dtype=DTYPE)
            fx[:, :k] = xp.asarray(f(Xc[:, :k].reshape(R * k, n)), dtype=DTYPE).reshape(R, k)
            nfe += k
            gi = xp.argmin(fx, axis=1)
            gv = xp.min(fx, axis=1)
            imp = gv < best_f
            best_x = xp.where(imp[:, None], _take_rows(Xc, gi), best_x)
            best_f = xp.where(imp, gv, best_f)
            gen_evals.append(nfe); gen_best.append(best_f.copy())
            if k < lam:                                  # budget exhausted mid-generation
                break
            g += 1
            order = xp.argsort(fx, axis=1)[:, :mu]
            ysel = xp.take_along_axis(y, order[..., None], axis=1)  # (R,mu,n)
            yw = xp.einsum("m,rmn->rn", W, ysel)
            m = m + sigma[:, None] * yw
            # C^{-1/2} yw = B diag(1/D) B^T yw
            invsq = xp.einsum("rij,rj,rkj,rk->ri", B, 1 / Dg, B, yw)
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * invsq
            psn = xp.linalg.norm(ps, axis=1)
            hsig = (psn / xp.sqrt(1 - (1 - cs) ** (2 * g)) / chiN) < (1.4 + 2 / (n + 1))
            hs = hsig.astype(DTYPE)
            pc = (1 - cc) * pc + hs[:, None] * math.sqrt(cc * (2 - cc) * mueff) * yw
            rank1 = pc[:, :, None] * pc[:, None, :]
            rankmu = xp.einsum("m,rmi,rmj->rij", W, ysel, ysel)
            dh = ((1 - hs) * cc * (2 - cc))[:, None, None]
            C = (1 - c1 - cmu) * C + c1 * (rank1 + dh * C) + cmu * rankmu
            sigma = sigma * xp.exp((cs / damps) * (psn / chiN - 1))

        curves, miles = _milestone_curves(gen_evals, gen_best, pop, iters, R)
        self.eval_axis = [np.full(R, mm) for mm in miles]
        return best_x, best_f, curves, nfe
