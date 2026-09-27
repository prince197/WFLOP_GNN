"""
============================================================
BATCHED OPTIMIZERS  (GPU or CPU)
============================================================

Each optimizer runs `n_runs` INDEPENDENT optimizations in
lockstep. Internally the population is an (n_runs, pop, dim)
tensor; every fitness evaluation flattens it to (n_runs*pop, dim)
and issues ONE batched objective call. With the default campaign
settings that is 30 seeds x 30 individuals = 900 layouts per call,
which is what makes a GPU worth using.

    best_x, best_f, curves, n_evals = ALGO(...).optimize(
        f_batch, dim, lb, ub, n_runs, pop, iters, seed)

    best_x  : (n_runs, dim)
    best_f  : (n_runs,)
    curves  : (n_runs, iters+1)
    n_evals : objective evaluations per run

FIDELITY TO THE ORIGINAL CPU CODE
---------------------------------
Identical update rules : GA, GWO, BBO, SSA, LX-SSA, QA-SSA
Changed, deliberately  : PSO and DE become SYNCHRONOUS
                         (generational) instead of asynchronous
                         (steady-state), and GWO now reports
                         best-so-far. See README_GPU.md section 4.

Random numbers are drawn in a different order from the original
per-individual loops, so a GPU run is statistically equivalent to
a CPU run, not bit-identical. Compare distributions over the 30
seeds, never single runs.
============================================================
"""

import math

import numpy as np

from backend import xp, DTYPE, asnumpy


# ---------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------
def _eval(f, X, counter):
    """X : (R,P,dim) -> (R,P) fitness, via one batched call."""
    r, p, d = X.shape
    vals = f(X.reshape(r * p, d))
    counter[0] += p
    return xp.asarray(vals, dtype=DTYPE).reshape(r, p)


def _eval_flat(f, X, counter, per_run=1):
    """X : (M,dim) -> (M,) fitness. `per_run` counts evals per run."""
    vals = f(X)
    counter[0] += per_run
    return xp.asarray(vals, dtype=DTYPE)


def _take_rows(A, idx):
    """A : (R,P,dim), idx : (R,) -> (R,dim)."""
    return xp.take_along_axis(A, idx[:, None, None], axis=1)[:, 0, :]


def _init(rng, n_runs, pop, dim, lb, ub):
    return xp.asarray(rng.uniform(lb, ub, (n_runs, pop, dim)), dtype=DTYPE)


class _Base:
    name = "base"

    def __init__(self, **kw):
        self.kw = kw
        self.eval_axis = []

    def _track(self, ne, n_runs, extra=None):
        """Record cumulative exact evaluations at this point of the curve.

        The convergence curve is indexed by iteration, but an efficiency claim
        has to be read against evaluations, not iterations - the algorithms
        spend very different amounts per iteration. Recording the count beside
        every curve point makes "best-so-far wake loss vs exact evaluations"
        a plot of stored data rather than a reconstruction.

        Stored per run, because the two gated variants genuinely spend
        different amounts in different runs.
        """
        v = float(ne[0])
        self.eval_axis.append(np.full(n_runs, v) if extra is None
                              else v + np.asarray(asnumpy(extra), dtype=float))

    def optimize(self, f, dim, lb, ub, n_runs, pop, iters, seed):
        raise NotImplementedError


# ===============================================================
# GA - identical semantics to the original
# ===============================================================
class GA(_Base):
    name = "GA"

    def optimize(self, f, dim, lb, ub, n_runs, pop, iters, seed):
        from backend import get_rng
        rng = get_rng(seed)
        pc = self.kw.get("pc", 0.8)
        pm = self.kw.get("pm", 0.1)
        ne = [0]

        X = _init(rng, n_runs, pop, dim, lb, ub)
        fit = _eval(f, X, ne)

        bidx = xp.argmin(fit, axis=1)
        best_x = _take_rows(X, bidx)
        best_f = xp.min(fit, axis=1)
        curves = [best_f.copy()]
        self._track(ne, n_runs)

        half = (pop + 1) // 2          # pairs of children; trimmed to pop below
        ar = xp.arange(n_runs)[:, None]

        for _ in range(iters):
            # --- binary tournament between two DISTINCT individuals ---
            parents = []
            for _p in range(2):
                i1 = xp.asarray(rng.integers(0, pop, (n_runs, half)))
                i2 = xp.asarray(rng.integers(0, pop - 1, (n_runs, half)))
                i2 = i2 + (i2 >= i1)                          # skip i1
                f1, f2 = fit[ar, i1], fit[ar, i2]
                win = xp.where(f1 < f2, i1, i2)
                parents.append(X[ar, win])                    # (R,half,dim)
            p1, p2 = parents

            # --- one-point crossover ---
            point = xp.asarray(rng.integers(1, dim, (n_runs, half)))[..., None]
            pos = xp.arange(dim)[None, None, :]
            before = pos < point
            do_x = (xp.asarray(rng.random((n_runs, half)))[..., None] < pc)

            c1 = xp.where(do_x, xp.where(before, p1, p2), p1)
            c2 = xp.where(do_x, xp.where(before, p2, p1), p2)
            child = xp.concatenate([c1, c2], axis=1)[:, :pop]  # (R,pop,dim)

            # --- uniform reset mutation ---
            mask = xp.asarray(rng.random(child.shape)) < pm
            fresh = xp.asarray(rng.uniform(lb, ub, child.shape), dtype=DTYPE)
            X = xp.clip(xp.where(mask, fresh, child), lb, ub)

            fit = _eval(f, X, ne)

            gi = xp.argmin(fit, axis=1)
            gv = xp.min(fit, axis=1)
            imp = gv < best_f
            best_x = xp.where(imp[:, None], _take_rows(X, gi), best_x)
            best_f = xp.where(imp, gv, best_f)
            curves.append(best_f.copy())
            self._track(ne, n_runs)

        return best_x, best_f, xp.stack(curves, axis=1), ne[0]


# ===============================================================
# PSO - SYNCHRONOUS variant (see README_GPU.md section 4)
# ===============================================================
class PSO(_Base):
    name = "PSO"

    def optimize(self, f, dim, lb, ub, n_runs, pop, iters, seed):
        from backend import get_rng
        rng = get_rng(seed)
        w = self.kw.get("w", 0.7)
        c1 = self.kw.get("c1", 2.0)
        c2 = self.kw.get("c2", 2.0)
        ne = [0]

        X = _init(rng, n_runs, pop, dim, lb, ub)
        V = xp.zeros_like(X)
        fit = _eval(f, X, ne)

        pbest, pbest_f = X.copy(), fit.copy()
        gi = xp.argmin(pbest_f, axis=1)
        gbest = _take_rows(pbest, gi)
        gbest_f = xp.min(pbest_f, axis=1)
        curves = [gbest_f.copy()]
        self._track(ne, n_runs)

        for _ in range(iters):
            r1 = xp.asarray(rng.random(X.shape), dtype=DTYPE)
            r2 = xp.asarray(rng.random(X.shape), dtype=DTYPE)

            V = (w * V
                 + c1 * r1 * (pbest - X)
                 + c2 * r2 * (gbest[:, None, :] - X))
            X = xp.clip(X + V, lb, ub)

            fit = _eval(f, X, ne)

            imp = fit < pbest_f
            pbest = xp.where(imp[..., None], X, pbest)
            pbest_f = xp.where(imp, fit, pbest_f)

            gi = xp.argmin(pbest_f, axis=1)
            gv = xp.min(pbest_f, axis=1)
            better = gv < gbest_f
            gbest = xp.where(better[:, None], _take_rows(pbest, gi), gbest)
            gbest_f = xp.where(better, gv, gbest_f)
            curves.append(gbest_f.copy())
            self._track(ne, n_runs)

        return gbest, gbest_f, xp.stack(curves, axis=1), ne[0]


# ===============================================================
# DE/rand/1/bin - SYNCHRONOUS variant (see README_GPU.md section 4)
# ===============================================================
class DE(_Base):
    name = "DE"

    def optimize(self, f, dim, lb, ub, n_runs, pop, iters, seed):
        from backend import get_rng
        rng = get_rng(seed)
        F = self.kw.get("F", 0.5)
        CR = self.kw.get("CR", 0.9)
        ne = [0]

        X = _init(rng, n_runs, pop, dim, lb, ub)
        fit = _eval(f, X, ne)

        best_f = xp.min(fit, axis=1)
        best_x = _take_rows(X, xp.argmin(fit, axis=1))
        curves = [best_f.copy()]
        self._track(ne, n_runs)

        ar = xp.arange(n_runs)[:, None]
        self_idx = xp.arange(pop)[None, :]

        for _ in range(iters):
            # three donors, mutually distinct and different from i,
            # by index-offsetting (same guarantee as np.random.choice
            # with replace=False in the original)
            i0 = self_idx
            ra = xp.asarray(rng.integers(0, pop - 1, (n_runs, pop)))
            ra = ra + (ra >= i0)

            s1 = xp.minimum(i0, ra)
            s2 = xp.maximum(i0, ra)
            rb = xp.asarray(rng.integers(0, pop - 2, (n_runs, pop)))
            rb = rb + (rb >= s1)
            rb = rb + (rb >= s2)

            t1 = xp.minimum(s1, rb)
            t3 = xp.maximum(s2, rb)
            t2 = i0 + ra + rb - t1 - t3
            rc = xp.asarray(rng.integers(0, pop - 3, (n_runs, pop)))
            rc = rc + (rc >= t1)
            rc = rc + (rc >= t2)
            rc = rc + (rc >= t3)

            a, b, c = X[ar, ra], X[ar, rb], X[ar, rc]

            mutant = xp.clip(a + F * (b - c), lb, ub)

            cross = xp.asarray(rng.random(X.shape)) < CR
            jrand = xp.asarray(rng.integers(0, dim, (n_runs, pop)))[..., None]
            cross = cross | (xp.arange(dim)[None, None, :] == jrand)

            trial = xp.clip(xp.where(cross, mutant, X), lb, ub)
            tfit = _eval(f, trial, ne)

            take = tfit < fit
            X = xp.where(take[..., None], trial, X)
            fit = xp.where(take, tfit, fit)

            gv = xp.min(fit, axis=1)
            imp = gv < best_f
            best_x = xp.where(imp[:, None], _take_rows(X, xp.argmin(fit, axis=1)), best_x)
            best_f = xp.where(imp, gv, best_f)
            curves.append(best_f.copy())
            self._track(ne, n_runs)

        return best_x, best_f, xp.stack(curves, axis=1), ne[0]


# ===============================================================
# GWO - update rule identical; best-so-far reporting FIXED
# ===============================================================
class GWO(_Base):
    name = "GWO"

    def optimize(self, f, dim, lb, ub, n_runs, pop, iters, seed):
        from backend import get_rng
        rng = get_rng(seed)
        legacy = self.kw.get("legacy_reporting", False)
        ne = [0]

        X = _init(rng, n_runs, pop, dim, lb, ub)
        fit = _eval(f, X, ne)

        best_f = xp.min(fit, axis=1)
        best_x = _take_rows(X, xp.argmin(fit, axis=1))
        curves = [best_f.copy()]
        self._track(ne, n_runs)

        for t in range(iters):
            order = xp.argsort(fit, axis=1)
            lead = [_take_rows(X, order[:, k]) for k in range(3)]

            a = 2.0 - 2.0 * t / iters
            acc = xp.zeros_like(X)
            for L in lead:
                r1 = xp.asarray(rng.random(X.shape), dtype=DTYPE)
                r2 = xp.asarray(rng.random(X.shape), dtype=DTYPE)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = xp.abs(C * L[:, None, :] - X)
                acc = acc + (L[:, None, :] - A * D)
            X = xp.clip(acc / 3.0, lb, ub)

            fit = _eval(f, X, ne)

            gv = xp.min(fit, axis=1)
            imp = gv < best_f
            best_x = xp.where(imp[:, None], _take_rows(X, xp.argmin(fit, axis=1)), best_x)
            best_f = xp.where(imp, gv, best_f)
            curves.append(best_f.copy())
            self._track(ne, n_runs)

        if legacy:                      # reproduce the original CPU behaviour
            gi = xp.argmin(fit, axis=1)
            return _take_rows(X, gi), xp.min(fit, axis=1), xp.stack(curves, axis=1), ne[0]
        return best_x, best_f, xp.stack(curves, axis=1), ne[0]


# ===============================================================
# BBO - identical semantics
# ===============================================================
class BBO(_Base):
    name = "BBO"

    def optimize(self, f, dim, lb, ub, n_runs, pop, iters, seed):
        from backend import get_rng
        rng = get_rng(seed)
        p_mut = self.kw.get("p_mutate", 0.01)
        keep = self.kw.get("keep", 2)
        ne = [0]

        X = _init(rng, n_runs, pop, dim, lb, ub)
        cost = _eval(f, X, ne)

        order = xp.argsort(cost, axis=1)
        X = xp.take_along_axis(X, order[..., None], axis=1)
        cost = xp.take_along_axis(cost, order, axis=1)
        curves = [cost[:, 0].copy()]
        self._track(ne, n_runs)

        ranks = xp.arange(1, pop + 1, dtype=DTYPE)
        mu = (pop + 1 - ranks) / (pop + 1)          # emigration, by rank
        lam = 1.0 - mu                              # immigration
        cdf = xp.cumsum(mu / xp.sum(mu))

        for _ in range(iters):
            elite_x = X[:, :keep].copy()
            elite_c = cost[:, :keep].copy()

            migrate = xp.asarray(rng.random((n_runs, pop, dim))) < lam[None, :, None]
            u = xp.asarray(rng.random((n_runs, pop, dim)), dtype=DTYPE)
            donor = xp.searchsorted(cdf, u.reshape(-1)).reshape(u.shape)
            donor = xp.clip(donor, 0, pop - 1)

            # take_along_axis on axis 1 gives exactly X[r, donor[r,k,j], j]
            gathered = xp.take_along_axis(X, donor, axis=1)

            X = xp.where(migrate, gathered, X)

            mmask = xp.asarray(rng.random((n_runs, pop, dim))) < p_mut
            fresh = xp.asarray(rng.uniform(lb, ub, (n_runs, pop, dim)), dtype=DTYPE)
            X = xp.clip(xp.where(mmask, fresh, X), lb, ub)

            cost = _eval(f, X, ne)

            order = xp.argsort(cost, axis=1)
            X = xp.take_along_axis(X, order[..., None], axis=1)
            cost = xp.take_along_axis(cost, order, axis=1)

            X[:, -keep:] = elite_x                  # elitism, as in the original
            cost[:, -keep:] = elite_c

            order = xp.argsort(cost, axis=1)
            X = xp.take_along_axis(X, order[..., None], axis=1)
            cost = xp.take_along_axis(cost, order, axis=1)

            curves.append(cost[:, 0].copy())
            self._track(ne, n_runs)

        return X[:, 0, :], cost[:, 0], xp.stack(curves, axis=1), ne[0]


# ===============================================================
# SSA family
# ===============================================================
class _SSAFamily(_Base):
    """Shared leader move; subclasses supply the follower rule."""

    def _leaders(self, rng, food, half, dim, lb, ub, c1, n_runs):
        r2 = xp.asarray(rng.random((n_runs, half, dim)), dtype=DTYPE)
        r3 = xp.asarray(rng.random((n_runs, half, dim)), dtype=DTYPE)
        step = c1 * ((ub - lb) * r2 + lb)
        return xp.where(r3 < 0.5,
                        food[:, None, :] + step,
                        food[:, None, :] - step)

    def optimize(self, f, dim, lb, ub, n_runs, pop, iters, seed):
        from backend import get_rng
        rng = get_rng(seed)
        ne = [0]

        X = _init(rng, n_runs, pop, dim, lb, ub)
        fit = _eval(f, X, ne)

        order = xp.argsort(fit, axis=1)
        X = xp.take_along_axis(X, order[..., None], axis=1)
        fit = xp.take_along_axis(fit, order, axis=1)

        food, food_f = X[:, 0, :].copy(), fit[:, 0].copy()
        curves = [food_f.copy()]
        self._track(ne, n_runs)

        half = pop // 2
        for t in range(1, iters + 1):
            c1 = 2.0 * math.exp(-((4.0 * t / iters) ** 2))

            newX = X.copy()
            newX[:, :half, :] = self._leaders(rng, food, half, dim, lb, ub, c1, n_runs)

            self._followers(f, rng, X, newX, food, food_f, fit, half, pop,
                            dim, lb, ub, ne, n_runs)

            X = xp.clip(newX, lb, ub)
            fit = _eval(f, X, ne)

            order = xp.argsort(fit, axis=1)
            X = xp.take_along_axis(X, order[..., None], axis=1)
            fit = xp.take_along_axis(fit, order, axis=1)

            imp = fit[:, 0] < food_f
            food = xp.where(imp[:, None], X[:, 0, :], food)
            food_f = xp.where(imp, fit[:, 0], food_f)
            curves.append(food_f.copy())
            self._track(ne, n_runs)

        return food, food_f, xp.stack(curves, axis=1), ne[0]


class SSA(_SSAFamily):
    name = "SSA"

    def _followers(self, f, rng, X, newX, food, food_f, fit, half, pop,
                   dim, lb, ub, ne, n_runs):
        # sequential chain: newX[i] depends on newX[i-1]
        for i in range(half, pop):
            newX[:, i, :] = (newX[:, i - 1, :] + X[:, i, :]) / 2.0


class LXSSA(_SSAFamily):
    """SSA + Laplace-perturbation follower with greedy selection."""
    name = "LXSSA"

    def _followers(self, f, rng, X, newX, food, food_f, fit, half, pop,
                   dim, lb, ub, ne, n_runs):
        phi = self.kw.get("phi", 0.0)
        chi = self.kw.get("chi", 1.0)

        for i in range(half, pop):
            follower = xp.clip((X[:, i, :] + newX[:, i - 1, :]) / 2.0, lb, ub)

            z = xp.asarray(rng.random((n_runs, 1)), dtype=DTYPE)
            gamma = xp.where(z <= 0.5,
                             phi + chi * xp.log(2 * z),
                             phi - chi * xp.log(2 * (1 - z)))

            cand = xp.clip(X[:, i, :] + gamma * (food - X[:, i, :]), lb, ub)

            both = xp.concatenate([cand, follower], axis=0)     # (2R,dim)
            vals = _eval_flat(f, both, ne, per_run=2)
            take = (vals[:n_runs] < vals[n_runs:])[:, None]
            newX[:, i, :] = xp.where(take, cand, follower)


class QASSA(_SSAFamily):
    """SSA + quadratic-interpolation follower with greedy selection."""
    name = "QASSA"

    def _followers(self, f, rng, X, newX, food, food_f, fit, half, pop,
                   dim, lb, ub, ne, n_runs):
        ar = xp.arange(n_runs)

        for i in range(half, pop):
            follower = xp.clip((X[:, i, :] + newX[:, i - 1, :]) / 2.0, lb, ub)

            # two distinct population members, neither of them the food source
            j1 = xp.asarray(rng.integers(1, pop, n_runs))
            j2 = xp.asarray(rng.integers(1, max(pop - 1, 2), n_runs))
            j2 = xp.where(j2 >= j1, j2 + 1, j2)
            j2 = xp.clip(j2, 1, pop - 1)

            Bp, Cp = X[ar, j1], X[ar, j2]
            fB, fC = fit[ar, j1][:, None], fit[ar, j2][:, None]
            fH = food_f[:, None]
            H = food

            num = ((Bp ** 2 - Cp ** 2) * fH
                   + (Cp ** 2 - H ** 2) * fB
                   + (H ** 2 - Bp ** 2) * fC)
            den = ((Bp - Cp) * fH
                   + (Cp - H) * fB
                   + (H - Bp) * fC)

            # relative guard: the original tested |den| < 1e-12 absolutely,
            # which is meaningless when fitness is ~1e30 inside penalties
            scale = xp.maximum(xp.abs(fH), 1.0)
            safe = xp.abs(den) > (1e-12 * scale)
            z = xp.where(safe, 0.5 * num / xp.where(safe, den, 1.0), H)
            cand = xp.clip(z, lb, ub)

            both = xp.concatenate([cand, follower], axis=0)
            vals = _eval_flat(f, both, ne, per_run=2)
            take = (vals[:n_runs] < vals[n_runs:])[:, None]
            newX[:, i, :] = xp.where(take, cand, follower)


ALGORITHMS = {
    "GA": GA, "PSO": PSO, "DE": DE, "GWO": GWO,
    "BBO": BBO, "SSA": SSA, "LXSSA": LXSSA, "QASSA": QASSA,
}

DEFAULT_KWARGS = {
    "GA": dict(pc=0.8, pm=0.1),
    "PSO": dict(w=0.7, c1=2.0, c2=2.0),
    "DE": dict(F=0.5, CR=0.9),
    "GWO": dict(),
    "BBO": dict(p_mutate=0.01, keep=2),
    "SSA": dict(),
    "LXSSA": dict(phi=0.0, chi=1.0),
    "QASSA": dict(),
}


def build(name, **overrides):
    """Construct an optimizer by name, with its published defaults.

    The GNN family lives in gnn_algorithms_gpu and registers itself on import.
    Importing it here, on demand, means build("GNNLXSSA") works from any entry
    point rather than only after the caller happens to have imported that
    module - the two names are otherwise indistinguishable to a caller.
    """
    if name not in ALGORITHMS and name.startswith("GNN"):
        import gnn_algorithms_gpu          # noqa: F401  (registers on import)
    if name not in ALGORITHMS:
        raise KeyError(f"unknown algorithm {name!r}; "
                       f"available: {', '.join(sorted(ALGORITHMS))}")
    kw = dict(DEFAULT_KWARGS.get(name, {}))
    kw.update(overrides)
    return ALGORITHMS[name](**kw)


# ===============================================================
# ACO  -  Eroglu & Seckiner (2012) continuous adaptation
# ---------------------------------------------------------------
# Faithful to the CPU class: pheromone = per-turbine leave-one-out
# contribution to the farm objective; ants are allocated to turbines in
# proportion to pheromone; each ant relocates its turbine and the move is
# kept only if the farm objective improves (greedy, sequential).
#
# Batched over RUNS only - the greedy accept chain inside a run is
# sequential by construction. The multinomial ant allocation is
# reproduced by drawing pop_size categorical samples and sorting them
# ascending, which is distributionally identical to the CPU's
# multinomial counts iterated over turbines in ascending order.
# ===============================================================
class ACO(_Base):
    name = "ACO"

    def optimize(self, f, dim, lb, ub, n_runs, pop, iters, seed):
        from backend import get_rng
        rng = get_rng(seed)
        rho = self.kw.get("rho", 0.1)
        p_global = self.kw.get("p_global", 0.5)
        sigma0 = self.kw.get("sigma0", 0.25)
        sigma_min = self.kw.get("sigma_min", 0.02)

        n = dim // 2
        radius = ub
        ne = [0]

        def sample_disk(shape):
            rad = radius * xp.sqrt(xp.asarray(rng.random(shape), dtype=DTYPE))
            ang = 2 * math.pi * xp.asarray(rng.random(shape), dtype=DTYPE)
            return xp.stack([rad * xp.cos(ang), rad * xp.sin(ang)], axis=-1)

        cur = sample_disk((n_runs, n)).reshape(n_runs, dim)
        cur_f = _eval_flat(f, cur, ne, per_run=1)

        best_x, best_f = cur.copy(), cur_f.copy()
        curves = [best_f.copy()]
        self._track(ne, n_runs)

        tau = xp.full((n_runs, n), 1.0 / n, dtype=DTYPE)
        ar = xp.arange(n_runs)

        # row i lists every turbine except i - used for the leave-one-out
        # pheromone. Integer gather: portable and contiguous on both backends.
        if n > 1:
            grid = xp.arange(n)[None, :].repeat(n, axis=0)
            drop_idx = grid[grid != xp.arange(n)[:, None]].reshape(n, n - 1)

        for t in range(1, iters + 1):
            # ---- pheromone: leave-one-out marginals, one batched call ----
            if n > 1:
                # n copies of the layout, each missing a different turbine
                P = cur.reshape(n_runs, n, 2)
                red = P[:, drop_idx]                                # (R,n,n-1,2)
                vals = _eval_flat(f, red.reshape(n_runs * n, 2 * (n - 1)),
                                  ne, per_run=n).reshape(n_runs, n)
                tau_new = xp.maximum(cur_f[:, None] - vals, 0.0)
                tot = xp.sum(tau_new, axis=1, keepdims=True)
                good = (tot > 1e-12) & xp.isfinite(tot)
                tau_new = xp.where(good, tau_new / xp.where(good, tot, 1.0),
                                   1.0 / n)
            else:
                tau_new = xp.full((n_runs, n), 1.0 / n, dtype=DTYPE)

            tau = (1.0 - rho) * tau + rho * tau_new
            s = xp.sum(tau, axis=1, keepdims=True)
            tau = xp.where(s > 1e-12, tau / xp.where(s > 1e-12, s, 1.0), 1.0 / n)

            # ---- allocate pop ants over turbines (ascending order) ----
            cdf = xp.cumsum(tau, axis=1)
            cdf = cdf / cdf[:, -1:]
            u = xp.asarray(rng.random((n_runs, pop)), dtype=DTYPE)
            slot = xp.sum(u[:, :, None] > cdf[:, None, :], axis=2)
            slot = xp.clip(slot, 0, n - 1)
            slot = xp.sort(slot, axis=1)

            sigma_t = (sigma0 + (sigma_min - sigma0) * (t / iters)) * radius

            # ---- ants act one at a time (greedy accept-if-improved) ----
            for a in range(pop):
                i = slot[:, a]
                glob = xp.asarray(rng.random(n_runs), dtype=DTYPE) < p_global
                new_g = sample_disk((n_runs,))
                loc = xp.stack([cur[ar, 2 * i], cur[ar, 2 * i + 1]], axis=-1)
                new_l = loc + xp.asarray(
                    rng.normal(0.0, 1.0, (n_runs, 2)), dtype=DTYPE) * sigma_t
                new = xp.clip(xp.where(glob[:, None], new_g, new_l), lb, ub)

                cand = cur.copy()
                cand[ar, 2 * i] = new[:, 0]
                cand[ar, 2 * i + 1] = new[:, 1]

                cf = _eval_flat(f, cand, ne, per_run=1)
                take = (cf < cur_f)[:, None]
                cur = xp.where(take, cand, cur)
                cur_f = xp.where(take[:, 0], cf, cur_f)

            imp = cur_f < best_f
            best_x = xp.where(imp[:, None], cur, best_x)
            best_f = xp.where(imp, cur_f, best_f)
            curves.append(best_f.copy())
            self._track(ne, n_runs)

        return best_x, best_f, xp.stack(curves, axis=1), ne[0]


# ===============================================================
# PF  -  Eroglu & Seckiner (2013) particle-filtering adaptation
# ---------------------------------------------------------------
# Predict (Gaussian propagation with annealed bandwidth) -> weight
# (elite-quantile observation y_k, forced non-decreasing, exponent
# normalised by the elite spread) -> systematic resampling.
# Fully batched: identical semantics to the CPU class.
# ===============================================================
class PF(_Base):
    name = "PF"

    def optimize(self, f, dim, lb, ub, n_runs, pop, iters, seed):
        from backend import get_rng
        rng = get_rng(seed)
        elite_frac = self.kw.get("elite_frac", 0.20)
        sigma0 = self.kw.get("sigma0", 0.20)
        sigma_min = self.kw.get("sigma_min", 0.005)
        beta = self.kw.get("beta", 5.0)
        ne = [0]

        span = ub - lb
        X = _init(rng, n_runs, pop, dim, lb, ub)
        fit = _eval(f, X, ne)

        best_f = xp.min(fit, axis=1)
        best_x = _take_rows(X, xp.argmin(fit, axis=1))
        curves = [best_f.copy()]
        self._track(ne, n_runs)

        y_prev = xp.full((n_runs,), -math.inf, dtype=DTYPE)
        n_elite = max(1, math.ceil(elite_frac * pop))

        for t in range(1, iters + 1):
            sigma_t = (sigma0 + (sigma_min - sigma0) * (t / iters)) * span
            X = xp.clip(X + xp.asarray(rng.normal(0.0, 1.0, X.shape),
                                       dtype=DTYPE) * sigma_t, lb, ub)
            fit = _eval(f, X, ne)

            gv = xp.min(fit, axis=1)
            imp = gv < best_f
            best_x = xp.where(imp[:, None], _take_rows(X, xp.argmin(fit, axis=1)),
                              best_x)
            best_f = xp.where(imp, gv, best_f)

            H = -fit
            srt = xp.sort(H, axis=1)
            y_k = srt[:, -n_elite]
            y_k = xp.maximum(y_k, y_prev)
            y_prev = y_k

            elite = H >= y_k[:, None]
            hmax = xp.max(xp.where(elite, H, -math.inf), axis=1)
            scale = hmax - y_k
            ok = xp.isfinite(scale) & (scale > 1e-12)

            expo = beta * (H - y_k[:, None]) / xp.where(ok, scale, 1.0)[:, None]
            w = xp.where(elite, xp.exp(xp.minimum(expo, 60.0)), 0.0)
            w = xp.where(ok[:, None], w, elite.astype(DTYPE))
            wsum = xp.sum(w, axis=1, keepdims=True)
            bad = ~(xp.isfinite(wsum) & (wsum > 1e-12))
            w = xp.where(bad, elite.astype(DTYPE), w)
            wsum = xp.sum(w, axis=1, keepdims=True)
            # last-resort uniform weights (all-NaN fitness for a run)
            w = xp.where(wsum > 0, w, 1.0 / pop)
            wsum = xp.sum(w, axis=1, keepdims=True)
            w = w / wsum

            # systematic resampling, per run
            u0 = xp.asarray(rng.random((n_runs, 1)), dtype=DTYPE)
            pos = (u0 + xp.arange(pop, dtype=DTYPE)[None, :]) / pop
            cum = xp.cumsum(w, axis=1)
            cum = cum / xp.maximum(cum[:, -1:], 1e-30)
            idx = xp.sum(pos[:, :, None] > cum[:, None, :], axis=2)
            idx = xp.clip(idx, 0, pop - 1)

            X = xp.take_along_axis(X, idx[..., None], axis=1).copy()
            curves.append(best_f.copy())
            self._track(ne, n_runs)

        return best_x, best_f, xp.stack(curves, axis=1), ne[0]


ALGORITHMS["ACO"] = ACO
ALGORITHMS["PF"] = PF
DEFAULT_KWARGS["ACO"] = dict(rho=0.1, p_global=0.5, sigma0=0.25, sigma_min=0.02)
DEFAULT_KWARGS["PF"] = dict(elite_frac=0.20, sigma0=0.20, sigma_min=0.005, beta=5.0)
