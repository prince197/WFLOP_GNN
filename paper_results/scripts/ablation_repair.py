"""
============================================================
ABLATION ARM:  LX-SSA + REPAIR, NO SURROGATE
============================================================

This module adds one optimizer, LXSSA_REPAIR, whose only purpose
is to sit between two arms the campaign already contains:

    LX-SSA          penalty constraint handling, no repair, no surrogate
    LXSSA_REPAIR    repair constraint handling, no surrogate      <- here
    GNN-LX-SSA      repair constraint handling, GNWM surrogate

The loop below is the `_GNNSalpBase.optimize` loop of
gnn_algorithms_gpu.py with every surrogate component deleted and
NOTHING else changed:

  * identical feasible initialization (feasible_layouts)
  * identical leader move and follower chain
  * identical Laplace candidate generation (Eq. 28)
  * identical boundary + spacing repair of BOTH candidate sets
  * identical incumbent rule (exact evaluations only)

The two deletions are:

  1. the guidance term.  GNN-LX-SSA forms
         cands = base + (tau * r) * ghat
     where ghat is the surrogate's direction head. With no surrogate
     there is no ghat, so cands = base.

  2. the screening step.  GNN-LX-SSA predicts all 2*pop candidates
     with the surrogate and evaluates only the best mu*2*pop of them
     exactly. With no surrogate there is nothing to screen with, so
     all 2*pop candidates are evaluated exactly.

Deletion (2) is what makes this arm cost 6,030 exact evaluations per
run - the same budget as LX-SSA, and about three times GNN-LX-SSA's.
That is the point: at LX-SSA's own budget it isolates the repair map,
and against GNN-LX-SSA it prices the surrogate.

No pre-training and no finite-difference labels are drawn, because
both exist solely to fit the network.
============================================================
"""

import numpy as np

from backend import xp, DTYPE, get_rng
from algorithms_gpu import _Base, _eval_flat, ALGORITHMS
from gnn_algorithms_gpu import (boundary_repair, spacing_repair,
                                feasible_layouts, _dataset_globals)
import objective_gpu as OBJ


class LXSSA_REPAIR(_Base):
    """LX-SSA under the repair map of Section 5.5, without the surrogate."""

    name = "LXSSA_REPAIR"

    def optimize(self, f, dim, lb, ub, n_runs, pop, iters, seed):
        rng = get_rng(seed)
        n = dim // 2
        r = float(ub)

        phi = self.kw.get("phi", 0.0)
        chi = self.kw.get("chi", 1.0)
        dataset = self.kw.get("dataset", 1)

        _omega, ideal, _psibar = _dataset_globals(dataset)
        min_dist = OBJ.min_spacing(dataset)

        ne = [0]
        self.n_surrogate_evals = 0
        self.n_surrogate_train = 0
        self.n_finetunes = 0

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
        self._track(ne, n_runs)

        # ---------------- main loop ----------------
        for t in range(1, iters + 1):
            r1 = 2.0 * float(np.exp(-((4.0 * t / iters) ** 2)))

            # ---- leader (Eq. 6) + follower chain ----
            moved = popX.copy()
            r2 = xp.asarray(rng.random((n_runs, dim)), dtype=DTYPE)
            r3 = xp.asarray(rng.random((n_runs, dim)), dtype=DTYPE)
            step = r1 * ((ub - lb) * r2 + lb)
            moved[:, 0, :] = xp.where(r3 >= 0.5, H + step, H - step)
            for i in range(1, pop):
                moved[:, i, :] = 0.5 * (popX[:, i, :] + moved[:, i - 1, :])

            # ---- Laplace candidates (Eq. 28); no guidance term ----
            z = xp.asarray(rng.random((n_runs, pop, 1)), dtype=DTYPE)
            gamma = xp.where(z <= 0.5,
                             phi - chi * xp.log(xp.maximum(z, 1e-12)),
                             phi + chi * xp.log(z))
            cands = moved + gamma * (H[:, None, :] - moved)

            # ---- repair both sets (identical to the GNN arm) ----
            mv = spacing_repair(boundary_repair(moved.reshape(n_runs, pop, n, 2), r),
                                r, passes=1, min_dist=min_dist)
            cd = spacing_repair(boundary_repair(cands.reshape(n_runs, pop, n, 2), r),
                                r, passes=1, min_dist=min_dist)
            moved = xp.clip(mv.reshape(n_runs, pop, dim), lb, ub)
            cands = xp.clip(cd.reshape(n_runs, pop, dim), lb, ub)

            allc = xp.concatenate([moved, cands], axis=1)          # (R,2P,dim)

            # ---- no screening: every candidate is evaluated exactly ----
            k = 2 * pop
            vals = _eval_flat(f, allc.reshape(n_runs * k, dim), ne,
                              per_run=k).reshape(n_runs, k)

            # ---- incumbent from exact values only ----
            gi = xp.argmin(vals, axis=1)
            gv = xp.min(vals, axis=1)
            imp = gv < best_f
            best_x = xp.where(imp[:, None],
                              xp.take_along_axis(allc, gi[:, None, None],
                                                 axis=1)[:, 0, :], best_x)
            best_f = xp.where(imp, gv, best_f)
            H = best_x.copy()

            # ---- next population: best `pop` of the 2*pop evaluated ----
            keep = xp.argsort(vals, axis=1)[:, :pop]
            popX = xp.take_along_axis(allc, keep[..., None], axis=1)
            fit = xp.take_along_axis(vals, keep, axis=1)

            curves.append(best_f.copy())
            self._track(ne, n_runs)

        total = xp.full(n_runs, float(ne[0]), dtype=DTYPE)
        self.n_exact_evals = total
        self.n_objective_calls = total
        return best_x, best_f, xp.stack(curves, axis=1), total


ALGORITHMS["LXSSA_REPAIR"] = LXSSA_REPAIR
