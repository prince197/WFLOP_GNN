"""
============================================================
GNWM SURROGATE - batched, dense, per-run weights (GPU or CPU)
============================================================

Batched port of the `_GNWMSurrogate` class in the CPU package: the
message-passing network of the GNN-LX-SSA paper (Sec. III-D2), with a
power head (Eq. 28), a direction head (Sec. III-D2b), manual backprop
and Adam.

WHAT CHANGED, AND WHY
---------------------
1. DENSE instead of sparse. The CPU version builds an explicit edge
   list (src, dst) per layout. Edge counts differ per layout, which
   cannot be batched. Here the graph is a dense (N, N) tensor with a
   boolean wake mask; messages from non-edges are multiplied by zero.
   Mathematically identical - a masked sum is the same as a sum over
   the edge list - but it batches.

2. PER-RUN WEIGHTS. Each of the R independent optimizations owns its
   own surrogate, exactly as on the CPU, so every weight carries a
   leading run axis and every matmul is a batched matmul.

3. ACTIVATIONS ARE RECOMPUTED IN THE BACKWARD PASS. The message
   tensor is (R, G, N, N, 2F+5); storing it for three layers would
   dominate memory. Only the layer inputs `h` are kept and the
   messages are recomputed - the standard activation-checkpointing
   trade-off, a little more compute for a lot less memory.

4. float32 BY DEFAULT for the surrogate only. The surrogate predicts
   wake loss as a percentage of ideal farm power (order 1-100), never
   the 1e30 penalty values, so single precision is ample and roughly
   doubles throughput. The exact objective stays float64.
   Override with WFLOP_SURROGATE_DTYPE=float64.

The arithmetic is otherwise the paper's: 7 node features (Eq. 24),
5 edge features (Eq. 25), three message-passing layers, hidden 64,
ReLU, Adam at lr 1e-3, joint loss L_power + lambda_g * L_dir.
============================================================
"""

import os
from backend import xp

SDTYPE = (xp.float64 if os.environ.get("WFLOP_SURROGATE_DTYPE", "float32")
          == "float64" else xp.float32)

NODE_F = 7
EDGE_F = 5


def _relu(z):
    return xp.maximum(z, 0)


def _bmm(x, W):
    """x : (R, ..., C)  W : (R, C, F)  ->  (R, ..., F)."""
    lead = x.shape[1:-1]
    flat = x.reshape(x.shape[0], -1, x.shape[-1])
    out = xp.matmul(flat, W)
    return out.reshape((x.shape[0],) + lead + (W.shape[-1],))


class BatchedGNWM:
    """R independent GNWM surrogates trained and evaluated in lockstep."""

    def __init__(self, n_runs, hidden=64, layers=3, seed=0, n_models=1):
        """n_models > 1 stacks a deep ensemble on the run axis
        (used by the _UQ variants); the effective leading dimension
        becomes R * n_models."""
        self.R = n_runs
        self.M = n_models
        self.RM = n_runs * n_models
        self.F = hidden
        self.L = layers

        from backend import get_rng
        rng = get_rng(seed)

        def xav(a, b):
            # standard_normal * sd: cupy.random.Generator has no normal()
            w = rng.standard_normal((self.RM, a, b)) * (2.0 / (a + b)) ** 0.5
            return xp.asarray(w, dtype=SDTYPE)

        def zeros(*shape):
            return xp.zeros((self.RM,) + shape, dtype=SDTYPE)

        F = hidden
        self.p = {"Win": xav(NODE_F, F), "bin": zeros(F),
                  "V1": xav(F, F), "d1": zeros(F),
                  "V2": xav(F, 1), "d2": zeros(1),
                  "G1": xav(F, F), "e1": zeros(F),
                  "G2": xav(F, 2), "e2": zeros(2)}
        for k in range(layers):
            self.p[f"W1{k}"] = xav(2 * F + EDGE_F, F)
            self.p[f"b1{k}"] = zeros(F)
            self.p[f"U1{k}"] = xav(2 * F, F)
            self.p[f"c1{k}"] = zeros(F)

        self.m = {k: xp.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: xp.zeros_like(v) for k, v in self.p.items()}
        self.t = xp.zeros((self.RM,), dtype=SDTYPE)   # Adam step, per row

    # -----------------------------------------------------------
    # forward
    # -----------------------------------------------------------
    def forward(self, Xn, Eattr, mask, need_cache=False):
        """Xn : (RM,G,N,7)   Eattr : (RM,G,N,N,5)   mask : (RM,G,N,N)
        returns P (RM,G), gout (RM,G,N,2), cache."""
        F = self.F
        maskf = mask.astype(SDTYPE)[..., None]

        z0 = _bmm(Xn, self.p["Win"]) + self.p["bin"][:, None, None, :]
        h = _relu(z0)

        cache = {"Xn": Xn, "z0": z0, "h": []} if need_cache else None

        for k in range(self.L):
            if need_cache:
                cache["h"].append(h)
            hi = xp.broadcast_to(h[:, :, :, None, :], h.shape[:3] + (h.shape[2], F))
            hj = xp.broadcast_to(h[:, :, None, :, :], h.shape[:3] + (h.shape[2], F))
            Min = xp.concatenate([hi, hj, Eattr], axis=-1)
            zm = _bmm(Min, self.p[f"W1{k}"]) + self.p[f"b1{k}"][:, None, None, None, :]
            m = _relu(zm) * maskf
            agg = xp.sum(m, axis=3)                       # sum over j

            Uin = xp.concatenate([h, agg], axis=-1)
            zu = _bmm(Uin, self.p[f"U1{k}"]) + self.p[f"c1{k}"][:, None, None, :]
            if need_cache:
                cache[f"zu{k}"] = zu
                cache[f"agg{k}"] = agg
            h = _relu(zu)

        z1 = _bmm(h, self.p["V1"]) + self.p["d1"][:, None, None, :]
        ph = _relu(z1)
        pnode = _bmm(ph, self.p["V2"]) + self.p["d2"][:, None, None, :]
        P = xp.sum(pnode[..., 0], axis=2)                 # (RM,G)

        zg = _bmm(h, self.p["G1"]) + self.p["e1"][:, None, None, :]
        hg = _relu(zg)
        gout = _bmm(hg, self.p["G2"]) + self.p["e2"][:, None, None, :]

        if need_cache:
            cache.update({"hL": h, "z1": z1, "ph": ph, "zg": zg, "hg": hg,
                          "gout": gout, "Eattr": Eattr, "maskf": maskf})
        return P, gout, cache

    def predict(self, Xn, Eattr, mask):
        return self.forward(Xn, Eattr, mask)[0]

    def guidance(self, Xn, Eattr, mask):
        P, gout, _ = self.forward(Xn, Eattr, mask)
        nrm = xp.sqrt(xp.sum(gout * gout, axis=-1, keepdims=True))
        return P, gout / xp.maximum(nrm, 1e-12)

    # -----------------------------------------------------------
    # backward  (mirrors the CPU class, recomputing messages)
    # -----------------------------------------------------------
    def backward(self, cache, dP, dG=None):
        """dP : (RM,G) dL/dP     dG : (RM,G,N,2) or None."""
        F = self.F
        g = {}
        Xn = cache["Xn"]
        n = Xn.shape[2]

        def acc(x, y):
            """sum over the graph axis -> (RM, C, F) style gradients."""
            xf = x.reshape(x.shape[0], -1, x.shape[-1])
            yf = y.reshape(y.shape[0], -1, y.shape[-1])
            return xp.matmul(xf.transpose(0, 2, 1), yf)

        # ---- power head
        dp = xp.ascontiguousarray(
            xp.broadcast_to(dP[:, :, None, None], cache["ph"].shape[:3] + (1,)))
        g["V2"] = acc(cache["ph"], dp)
        g["d2"] = xp.sum(dp, axis=(1, 2))
        dph = _bmm(dp, self.p["V2"].transpose(0, 2, 1))
        dz1 = dph * (cache["z1"] > 0)
        g["V1"] = acc(cache["hL"], dz1)
        g["d1"] = xp.sum(dz1, axis=(1, 2))
        dh = _bmm(dz1, self.p["V1"].transpose(0, 2, 1))

        # ---- direction head
        if dG is not None:
            g["G2"] = acc(cache["hg"], dG)
            g["e2"] = xp.sum(dG, axis=(1, 2))
            dhg = _bmm(dG, self.p["G2"].transpose(0, 2, 1))
            dzg = dhg * (cache["zg"] > 0)
            g["G1"] = acc(cache["hL"], dzg)
            g["e1"] = xp.sum(dzg, axis=(1, 2))
            dh = dh + _bmm(dzg, self.p["G1"].transpose(0, 2, 1))
        else:
            for k in ("G1", "e1", "G2", "e2"):
                g[k] = xp.zeros_like(self.p[k])

        # ---- trunk
        for k in reversed(range(self.L)):
            h = cache["h"][k]
            dzu = dh * (cache[f"zu{k}"] > 0)
            Uin = xp.concatenate([h, cache[f"agg{k}"]], axis=-1)
            g[f"U1{k}"] = acc(Uin, dzu)
            g[f"c1{k}"] = xp.sum(dzu, axis=(1, 2))
            dUin = _bmm(dzu, self.p[f"U1{k}"].transpose(0, 2, 1))
            dh_prev = dUin[..., :F]
            dagg = dUin[..., F:]

            # recompute the messages of this layer
            hi = xp.broadcast_to(h[:, :, :, None, :], h.shape[:3] + (n, F))
            hj = xp.broadcast_to(h[:, :, None, :, :], h.shape[:3] + (n, F))
            Min = xp.concatenate([hi, hj, cache["Eattr"]], axis=-1)
            zm = _bmm(Min, self.p[f"W1{k}"]) + self.p[f"b1{k}"][:, None, None, None, :]

            dm = xp.broadcast_to(dagg[:, :, :, None, :], zm.shape) * cache["maskf"]
            dzm = dm * (zm > 0)
            g[f"W1{k}"] = acc(Min, dzm)
            g[f"b1{k}"] = xp.sum(dzm, axis=(1, 2, 3))
            dMin = _bmm(dzm, self.p[f"W1{k}"].transpose(0, 2, 1))

            dh_prev = dh_prev + xp.sum(dMin[..., :F], axis=3)      # dst = i
            dh_prev = dh_prev + xp.sum(dMin[..., F:2 * F], axis=2)  # src = j
            dh = dh_prev

        dz0 = dh * (cache["z0"] > 0)
        g["Win"] = acc(Xn, dz0)
        g["bin"] = xp.sum(dz0, axis=(1, 2))
        return g

    # -----------------------------------------------------------
    def adam(self, grads, lr=1e-3, active=None):
        """Adam step. `active` (RM,) bool updates only those rows (weights,
        moments and step count); the others are left exactly as they were,
        so each run's surrogate is trained only when its own retraining is
        due - as the CPU code does with one surrogate per run."""
        act = (xp.ones((self.RM,), dtype=bool) if active is None
               else active.astype(bool))
        self.t = self.t + act.astype(SDTYPE)
        tt = xp.maximum(self.t, 1.0)
        b1c = 1 - 0.9 ** tt
        b2c = 1 - 0.999 ** tt
        for k in self.p:
            gk = grads[k]
            shp = (self.RM,) + (1,) * (gk.ndim - 1)
            a = act.reshape(shp)
            m = 0.9 * self.m[k] + 0.1 * gk
            v = 0.999 * self.v[k] + 0.001 * gk * gk
            p = self.p[k] - lr * (m / b1c.reshape(shp)) / (
                xp.sqrt(v / b2c.reshape(shp)) + 1e-8)
            self.m[k] = xp.where(a, m, self.m[k])
            self.v[k] = xp.where(a, v, self.v[k])
            self.p[k] = xp.where(a, p, self.p[k]).astype(SDTYPE)

    def train(self, Xn, Eattr, mask, target, gfd=None, epochs=1, batch=16,
              lr=1e-3, lambda_g=0.5, rng=None, valid=None, active=None):
        """One joint training pass. Shapes carry the sample axis in G:
            Xn (RM,S,N,7)  Eattr (RM,S,N,N,5)  mask (RM,S,N,N)
            target (RM,S)  gfd (RM,S,N,2) or None (NaN rows = no label)
            valid (RM,S) bool - samples to train on. Samples outside it
            (unlabelled, or penalty-contaminated) contribute NOTHING to
            the loss, matching the CPU code, which simply never puts
            them in the replay buffer.
            active (RM,) bool or None - rows to update (None = all).
        """
        S = Xn.shape[1]
        if S == 0:
            return
        if not hasattr(self, "n_train_forwards"):
            self.n_train_forwards = 0
        from backend import get_rng
        rng = rng or get_rng(0)

        for _ in range(epochs):
            # Independent sample order per row, so ensemble members are trained
            # on the same data in different orders - the CPU code achieves this
            # by giving each member its own generator.
            order = xp.argsort(xp.asarray(rng.random((self.RM, S))), axis=1)

            for b0 in range(0, S, batch):
                idx = order[:, b0:b0 + batch]
                nb = int(idx.shape[1])
                xb = xp.take_along_axis(Xn, idx[:, :, None, None], axis=1)
                eb = xp.take_along_axis(Eattr, idx[:, :, None, None, None], axis=1)
                mb = xp.take_along_axis(mask, idx[:, :, None, None], axis=1)
                tb = xp.take_along_axis(target, idx, axis=1)

                P, gout, cache = self.forward(xb, eb, mb, need_cache=True)
                # Training forward passes are counted separately from
                # inference: they are a cost of building the surrogate, not a
                # substitute for an exact evaluation, and conflating the two
                # inflates the apparent surrogate load.
                self.n_train_forwards += nb * (
                    self.RM if active is None else int(xp.sum(active)))
                if valid is None:
                    wb = xp.ones_like(P)
                else:
                    wb = xp.take_along_axis(valid, idx, axis=1).astype(SDTYPE)
                cnt = xp.maximum(xp.sum(wb, axis=1, keepdims=True), 1.0)
                tb = xp.where(wb > 0, tb, P)                 # zero residual
                dP = 2.0 * (P - tb) * wb / cnt               # Eq. (30)

                dG = None
                if gfd is not None:
                    gb = xp.take_along_axis(gfd, idx[:, :, None, None], axis=1)
                    has = ~xp.any(xp.isnan(gb), axis=(2, 3))  # (RM,nb)
                    if bool(xp.any(has)):
                        gb = xp.where(xp.isnan(gb), 0.0, gb)
                        vn = xp.maximum(
                            xp.sqrt(xp.sum(gout * gout, axis=-1, keepdims=True)),
                            1e-9)
                        vhat = gout / vn
                        dot = xp.sum(gb * vhat, axis=-1, keepdims=True)
                        n_nodes = gout.shape[2]
                        dG = (lambda_g / (n_nodes * nb)) * (-gb + dot * vhat) / vn
                        dG = dG * (has.astype(SDTYPE)
                                   * wb)[:, :, None, None]

                grads = self.backward(cache, dP, dG)
                self.adam(grads, lr, active=active)


# ===============================================================
# GRAPH CONSTRUCTION  (Eqs. 24-25), batched and dense
# ===============================================================
def build_graphs(P, r, omega, thetas, alpha_cone, aj, wt_k, wt_r, psibar):
    """P : (B,G,N,2) positions -> (Xn, Eattr, mask) for the surrogate.

    Mirrors _GNWMSurrogate.build_graph: direction-aggregated wake mask
    over the 24 bins, five edge features, seven node features.
    """
    B, G, N, _ = P.shape
    RK = wt_r / wt_k

    dx = P[..., 0][..., :, None] - P[..., 0][..., None, :]      # (B,G,N,N)
    dy = P[..., 1][..., :, None] - P[..., 1][..., None, :]

    sumw = xp.zeros((B, G, N, N), dtype=SDTYPE)
    dbar = xp.zeros_like(sumw)
    bbar = xp.zeros_like(sumw)
    sbar = xp.zeros_like(sumw)

    eye = xp.eye(N, dtype=bool)[None, None]
    for l in range(thetas.shape[0]):
        w = float(omega[l])
        if w <= 0:
            continue
        th = float(thetas[l])
        ct, st = xp.cos(xp.asarray(th)), xp.sin(xp.asarray(th))
        proj = dx * ct + dy * st
        d = xp.abs(proj)
        den = xp.maximum(xp.sqrt((dx + RK * ct) ** 2 + (dy + RK * st) ** 2), 1e-12)
        beta = xp.arccos(xp.clip((proj + RK) / den, -1, 1))
        msk = (beta < alpha_cone) & (~eye)
        mf = msk.astype(SDTYPE)
        defc = aj / (1.0 + (wt_k / wt_r) * d) ** 2
        sumw = sumw + w * mf
        dbar = dbar + w * mf * d.astype(SDTYPE)
        bbar = bbar + w * mf * beta.astype(SDTYPE)
        sbar = sbar + w * mf * defc.astype(SDTYPE)

    mask = sumw > 0
    sw = xp.where(mask, sumw, 1.0)

    Eattr = xp.stack([
        dbar / sw / r,
        bbar / sw / alpha_cone,
        (dx / r).astype(SDTYPE),
        (dy / r).astype(SDTYPE),
        sbar / sw,
    ], axis=-1) * mask.astype(SDTYPE)[..., None]

    edir_x = float(xp.sum(omega * xp.cos(thetas)))
    edir_y = float(xp.sum(omega * xp.sin(thetas)))
    ones = xp.ones((B, G, N), dtype=SDTYPE)
    Xn = xp.stack([
        (P[..., 0] / r).astype(SDTYPE),
        (P[..., 1] / r).astype(SDTYPE),
        ((P[..., 0] ** 2 + P[..., 1] ** 2) / r ** 2).astype(SDTYPE),
        ones,
        ones * (psibar / 13.0),
        ones * edir_x,
        ones * edir_y,
    ], axis=-1)
    return Xn, Eattr, mask
