"""
============================================================
BATCHED WFLOP OBJECTIVE  (GPU or CPU)
============================================================

Same physics as the original objective.py - Jensen wake model,
24-sector Weibull expected power, boundary and spacing penalties -
but evaluating B candidate layouts in ONE call, with no Python
loop over layouts, turbines, sectors or speed bins.

    objective_batch(X, farm_radius) -> (B,) objective values
        X : (B, 2N) array of flattened layouts

That batching is the whole point. A single layout is far too small
to occupy a GPU: for N = 18 the wake tensor is 24*18*18 = 7,776
elements, less than one kernel launch is worth. With B = 900
(30 seeds x 30 individuals evaluated together) the same tensor is
7 million elements, which is a real workload.

Numerically this reproduces objective.py exactly on the CPU
backend - verified element-wise by validate_gpu.py.
============================================================
"""

import math

from backend import xp, DTYPE

# ---------------------------------------------------------------
# WAKE MODEL CONSTANTS   (identical to objective.py)
# ---------------------------------------------------------------
R = 38.5
K = 0.075
CT = 0.8

A = 1 - (1 - CT) ** 0.5
B_COEF = K / R
ALPHA = float(xp.arctan(xp.asarray(K, dtype=xp.float64)))

PENALTY = 1e10
MIN_SPACING = 8 * R

# ---------------------------------------------------------------
# POWER MODEL CONSTANTS  (identical to objective.py)
# ---------------------------------------------------------------
IDEAL_POWER_SCEN1 = 14045.7374

_OMEGA = [0, 0.01, 0.01, 0.01, 0.01,
          0.20,
          0.60,
          0.01, 0.01, 0.01, 0.01,
          0.01, 0.01, 0.01, 0.01,
          0.01, 0.01, 0.01, 0.01,
          0.01, 0.01, 0.01, 0.01,
          0]

_SPEED = [3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9,
          9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14]

K_SHAPE = 2
LAMBDA = 140.86
ETTA = -500.0
P_RATED = 1500.0
CUT_IN = 3.5
RATED = 14.0

BASE_SPEED = 13.0
SECTOR_WIDTH = 15.0          # degrees; also the weight scale in expected_power

# ---- precomputed device-side constants -------------------------
OMEGA = xp.asarray(_OMEGA, dtype=DTYPE)                     # (24,)
SPEED = xp.asarray(_SPEED, dtype=DTYPE)                     # (22,)
W_SECTOR = (SECTOR_WIDTH * OMEGA)[None, :, None]            # (1,24,1)
SPEED_MID = ((SPEED[:-1] + SPEED[1:]) / 2)[:, None, None]         # (21,1,1)

# ---------------------------------------------------------------
# WIND DATA SETS  (identical constants to objective.py v34)
# ---------------------------------------------------------------
_OMEGA2 = [0.0002, 0.0080, 0.0227, 0.0242, 0.0225, 0.0339, 0.0423, 0.0290,
           0.0617, 0.0813, 0.0994, 0.1394, 0.1839, 0.1115, 0.0765, 0.0080,
           0.0051, 0.0019, 0.0012, 0.0010, 0.0017, 0.0031, 0.0097, 0.0317]
_PSI2 = [7.0, 5.0, 5.0, 5.0, 5.0, 4.0, 5.0, 6.0, 7.0, 7.0, 8.0, 9.5,
         10.0, 8.5, 8.5, 6.5, 4.6, 2.6, 8.0, 5.0, 6.4, 5.2, 4.5, 3.9]

OMEGA_1 = xp.asarray(_OMEGA, dtype=DTYPE)
PSI_1 = xp.full((24,), BASE_SPEED, dtype=DTYPE)
OMEGA_2 = xp.asarray(_OMEGA2, dtype=DTYPE)
PSI_2 = xp.asarray(_PSI2, dtype=DTYPE)

# NOTE ON THE DIRECTION WEIGHTS. Data Set I sums to exactly 1.0. Data Set II
# sums to 0.9999, not 1.0 - that is the published table, reproduced verbatim
# from the source implementation, NOT a transcription error and NOT something
# to "fix" by renormalising. Renormalising would change every Data Set II
# result by a factor of 1.0001 and break comparability with the literature.
# The 1e-4 shortfall is absorbed into the wake loss and is far below the
# differences the campaign measures.
assert abs(float(sum(_OMEGA)) - 1.0) < 1e-12
assert abs(float(sum(_OMEGA2)) - 0.9999) < 1e-12

_dirs = [d + 7.5 for d in range(0, 360, int(SECTOR_WIDTH))]
THETA = xp.asarray(_dirs, dtype=DTYPE) * (math.pi / 180.0)  # (24,)
COS_T = xp.cos(THETA)[None, :, None, None]                  # (1,24,1,1)
SIN_T = xp.sin(THETA)[None, :, None, None]                  # (1,24,1,1)


# ===============================================================
# WAKE MODEL SELECTION
# ===============================================================
# WFLOP_WAKE=jensen (default) keeps the benchmark's Jensen top-hat wake.
# WFLOP_WAKE=gaussian uses the Bastankhah & Porte-Agel (2014) Gaussian wake,
# with the wake-growth rate of Niayifar & Porte-Agel (2016),
#     k* = 0.3837 * TI + 0.003678,   TI from WFLOP_TI (default 0.075),
# the same thrust coefficient CT, the same Katic root-sum-square
# superposition and the same Weibull expected-power integration. Only the
# per-pair velocity deficit changes.
WAKE_MODEL = __import__("os").environ.get("WFLOP_WAKE", "jensen").lower()
if WAKE_MODEL not in ("jensen", "gaussian"):
    raise ValueError(f"WFLOP_WAKE must be 'jensen' or 'gaussian', not {WAKE_MODEL!r}")
TI = float(__import__("os").environ.get("WFLOP_TI", 0.075))
D_ROTOR = 2.0 * R
K_STAR = 0.3837 * TI + 0.003678
_BETA_G = 0.5 * (1.0 + (1.0 - CT) ** 0.5) / (1.0 - CT) ** 0.5
EPS_G = 0.2 * _BETA_G ** 0.5


def _gaussian_deficit(proj, lat2):
    """Bastankhah & Porte-Agel (2014) deficit for downstream separation `proj`
    (m, > 0 means downstream) and squared lateral offset `lat2` (m^2).
    The square-root argument is clipped at 0 in the near wake (x < ~2D),
    where the self-similar solution is not defined."""
    s = K_STAR * xp.maximum(proj, 0.0) / D_ROTOR + EPS_G          # sigma / D
    core = 1.0 - xp.sqrt(xp.clip(1.0 - CT / (8.0 * s * s), 0.0, 1.0))
    return core * xp.exp(-lat2 / (2.0 * (s * D_ROTOR) ** 2))


# ===============================================================
# WAKED WIND SPEED
# ===============================================================
def waked_speeds(P, psi=None):
    if WAKE_MODEL == "gaussian":
        return waked_speeds_gaussian(P, psi)
    return waked_speeds_jensen(P, psi)


def waked_speeds_gaussian(P, psi=None):
    """Gaussian-wake counterpart of waked_speeds_jensen: (B,N,2) -> (B,24,N)."""
    x = P[:, None, :, 0]
    y = P[:, None, :, 1]
    dx = x[..., :, None] - x[..., None, :]     # (B,1,N,N) : i - j
    dy = y[..., :, None] - y[..., None, :]
    proj = dx * COS_T + dy * SIN_T             # (B,24,N,N) downstream separation of i from j
    lat2 = xp.maximum(dx * dx + dy * dy - proj * proj, 0.0)
    n = P.shape[1]
    eye = xp.eye(n, dtype=bool)[None, None, :, :]
    downstream = (proj > 0.0) & (~eye)
    deficit = _gaussian_deficit(proj, lat2)
    total = xp.sqrt(xp.sum(xp.where(downstream, deficit * deficit, 0.0), axis=-1))
    total = xp.minimum(total, 0.999)          # guard: only binds for coincident (infeasible) layouts
    base = PSI_1 if psi is None else psi
    return base[None, :, None] * (1.0 - total)


def waked_speeds_jensen(P, psi=None):
    """P : (B, N, 2) -> (B, 24, N) waked wind speed per sector.

    `psi` is the per-sector free-stream Weibull scale (Data Set II and III
    make it direction-dependent); Data Set I uses a constant 13 m/s.
    """
    x = P[:, None, :, 0]                       # (B,1,N)
    y = P[:, None, :, 1]

    dx = x[..., :, None] - x[..., None, :]     # (B,1,N,N) : i - j
    dy = y[..., :, None] - y[..., None, :]

    proj = dx * COS_T + dy * SIN_T             # (B,24,N,N) downwind separation
    dist = xp.abs(proj)

    ex = dx + (R / K) * COS_T
    ey = dy + (R / K) * SIN_T
    denom = xp.sqrt(ex * ex + ey * ey)

    arg = xp.clip((proj + R / K) / denom, -1.0, 1.0)
    beta = xp.arccos(arg)

    deficit = A / ((1.0 + B_COEF * dist) ** 2)

    n = P.shape[1]
    eye = xp.eye(n, dtype=bool)[None, None, :, :]
    in_wake = (beta < ALPHA) & (~eye)

    # Katic root-sum-square superposition
    total = xp.sqrt(xp.sum(xp.where(in_wake, deficit * deficit, 0.0), axis=-1))
    base = PSI_1 if psi is None else psi        # (24,)
    return base[None, :, None] * (1.0 - total)  # (B,24,N)


# ===============================================================
# EXPECTED POWER
# ===============================================================
MEM_BUDGET_BYTES = int(float(__import__("os").environ.get("WFLOP_MEM_BUDGET_MB", 256)) * 2**20)


def _max_rows(n):
    """Largest batch whose wake tensors fit inside MEM_BUDGET_BYTES.

    waked_speeds() materialises several (B, 24, N, N) intermediates - the
    projection, distance, denominator, angle and deficit. That tensor is not
    chunkable inside the function, so the BATCH is split instead. Without this
    a large call (the finite-difference labels of the GNN pre-training build
    one) allocates tens of GB and the process is killed.
    """
    item = xp.dtype(DTYPE).itemsize
    per_row = 24 * n * n * item * 6          # ~6 live intermediates
    return max(1, int(MEM_BUDGET_BYTES // max(per_row, 1)))


def _auto_chunk(c):
    """Speed-bin chunk size that keeps the (nbins,B,24,N) tensor inside
    MEM_BUDGET_BYTES. Returns None when the full tensor already fits."""
    b, s_, n = c.shape
    item = xp.dtype(DTYPE).itemsize
    per_bin = b * s_ * n * item * 3          # E, band and one temporary
    full = per_bin * SPEED.shape[0]
    if full <= MEM_BUDGET_BYTES:
        return None
    return max(1, int(MEM_BUDGET_BYTES // per_bin))


def expected_power_batch(c, chunk=None, omega=None):
    """c : (B, 24, N) waked speeds -> (B,) expected power.

    `chunk` splits the speed-bin axis to bound peak memory; the
    full tensor is (22, B, 24, N), which for B = 900 and N = 18 is
    ~68 MB in float64. Chunking is off by default and only needed
    for very large batches on small cards.
    """
    inv = 1.0 / c                                        # (B,24,N)
    w_sector = (W_SECTOR if omega is None
                else (SECTOR_WIDTH * omega)[None, :, None])

    if chunk is None:
        chunk = _auto_chunk(c)

    if chunk is None:
        E = xp.exp(-((SPEED[:, None, None, None] * inv[None]) ** K_SHAPE))
        band = (w_sector[None] * (E[:-1] - E[1:]))       # (21,B,24,N)
        first = xp.sum(SPEED_MID * xp.sum(band, axis=2), axis=0)   # (B,N)
    else:
        first = xp.zeros((c.shape[0], c.shape[2]), dtype=c.dtype)     # (B,N)
        n_bins = SPEED.shape[0] - 1                       # 21 bins
        for s0 in range(0, n_bins, chunk):
            s1 = min(s0 + chunk, n_bins)
            E = xp.exp(-((SPEED[s0:s1 + 1][:, None, None, None]
                          * inv[None]) ** K_SHAPE))
            band = w_sector[None] * (E[:-1] - E[1:])      # (s1-s0,B,24,N)
            first = first + xp.sum(SPEED_MID[s0:s1] * xp.sum(band, axis=2), axis=0)

    e_rated = xp.exp(-((RATED * inv) ** K_SHAPE))
    e_cutin = xp.exp(-((CUT_IN * inv) ** K_SHAPE))

    second = xp.sum(w_sector * e_rated, axis=1)          # (B,N)
    third = xp.sum(w_sector * (e_cutin - e_rated), axis=1)

    per_turbine = LAMBDA * first + P_RATED * second + ETTA * third
    return xp.sum(per_turbine, axis=1)                   # (B,)


# ===============================================================
# POWER CURVE SELECTION
# ===============================================================
# WFLOP_POWER=linear (default) keeps the benchmark's linearised ramp above.
# WFLOP_POWER=ge15 uses the tabulated GE 1.5 MW / 77 m power curve of
# power_ge15.py (NREL turbine-models), integrated over the Weibull
# distribution by the midpoint rule on 0.25 m/s bins from cut-in to cut-out
# (zero above cut-out). The weights, the 15-degree scaling and hence the
# working units are those of the linear model, so only the curve changes.
# The ideal power of both data sets becomes the isolated-turbine value
# under this curve.
POWER_CURVE = __import__("os").environ.get("WFLOP_POWER", "linear").lower()
if POWER_CURVE not in ("linear", "ge15"):
    raise ValueError(f"WFLOP_POWER must be 'linear' or 'ge15', not {POWER_CURVE!r}")

if POWER_CURVE == "ge15":
    import power_ge15 as _PC
    _PC_EDGES = xp.asarray(_PC.EDGES, dtype=DTYPE)                     # (87,)
    _PC_P = xp.asarray(_PC.P_MID, dtype=DTYPE)[:, None, None]          # (86,1,1)

    def expected_power_batch(c, chunk=None, omega=None):   # noqa: F811
        """c : (B, 24, N) waked speeds -> (B,) expected power, GE 1.5 MW curve."""
        inv = 1.0 / c
        w_sector = (W_SECTOR if omega is None
                    else (SECTOR_WIDTH * omega)[None, :, None])
        if chunk is None:
            chunk = _auto_chunk(c)
        n_bins = _PC_EDGES.shape[0] - 1
        step = n_bins if chunk is None else max(1, int(chunk))
        per_turbine = xp.zeros((c.shape[0], c.shape[2]), dtype=c.dtype)
        for s0 in range(0, n_bins, step):
            s1 = min(s0 + step, n_bins)
            E = xp.exp(-((_PC_EDGES[s0:s1 + 1][:, None, None, None] * inv[None]) ** K_SHAPE))
            band = w_sector[None] * (E[:-1] - E[1:])                   # (s1-s0,B,24,N)
            per_turbine = per_turbine + xp.sum(_PC_P[s0:s1] * xp.sum(band, axis=2), axis=0)
        return xp.sum(per_turbine, axis=1)

    IDEAL_POWER_SCEN1 = float(expected_power_batch(
        PSI_1.reshape(1, 24, 1), omega=OMEGA_1)[0])


# ===============================================================
# CONSTRAINT PENALTIES
# ===============================================================
def boundary_penalty_batch(P, farm_radius):
    g = P[:, :, 0] ** 2 + P[:, :, 1] ** 2 - farm_radius ** 2
    viol = xp.maximum(g, 0.0)
    term = xp.where(g > 0, (1.0 + PENALTY * viol) ** 2, 0.0)
    return xp.sum(term, axis=1)


def spacing_penalty_batch(P):
    d = P[:, :, None, :] - P[:, None, :, :]
    dist = xp.sqrt(xp.sum(d * d, axis=-1))               # (B,N,N)
    g = MIN_SPACING - dist

    n = P.shape[1]
    iu = xp.triu(xp.ones((n, n), dtype=bool), k=1)[None]  # upper triangle only
    active = (g > 0) & iu

    term = xp.where(active, (1.0 + PENALTY * xp.where(active, g, 0.0)) ** 2, 0.0)
    return xp.sum(term, axis=(1, 2))


# ===============================================================
# OBJECTIVE
# ===============================================================
# Data Set II ideal power: isolated turbine under the same omega/psi, so a
# widely spaced layout gives ~0 wake loss (matches objective.py v34).
IDEAL_POWER_SCEN2 = float(expected_power_batch(
    PSI_2.reshape(1, 24, 1), omega=OMEGA_2)[0])

DATASETS = {
    1: dict(omega=OMEGA_1, psi=PSI_1, ideal=IDEAL_POWER_SCEN1),
    2: dict(omega=OMEGA_2, psi=PSI_2, ideal=IDEAL_POWER_SCEN2),
}


def _cfg(dataset):
    ds = int(dataset)
    if ds not in DATASETS:
        raise ValueError(
            f"unknown wind data set {ds}: this package provides 1 and 2 "
            "(Horns Rev 1 / Data Set III was removed)")
    return DATASETS[ds]


def objective_batch(X, farm_radius, chunk=None, dataset=1):
    """X : (B, 2N) flattened layouts -> (B,) objective values.

    dataset 1 : Wind Data Set I   (psi = 13 for every direction)
    dataset 2 : Wind Data Set II  (direction-dependent psi, Table II(b))
    """
    cfg = _cfg(dataset)

    X = xp.asarray(X, dtype=DTYPE)
    if X.ndim == 1:
        X = X[None, :]
    b = X.shape[0]
    n = X.shape[1] // 2

    # Split the batch when the wake tensors would not fit; the result is
    # identical, only the peak allocation changes.
    rows = _max_rows(n)
    if b > rows:
        return xp.concatenate([
            objective_batch(X[i:i + rows], farm_radius, chunk=chunk,
                            dataset=dataset)
            for i in range(0, b, rows)])

    P = X.reshape(b, n, 2)

    c = waked_speeds(P, cfg["psi"])
    ep = expected_power_batch(c, chunk=chunk, omega=cfg["omega"])

    loss = cfg["ideal"] * n - ep
    loss = loss + boundary_penalty_batch(P, farm_radius)
    loss = loss + spacing_penalty_batch(P)
    return loss


def energy_production_batch(X, dataset=1):
    """Expected energy production only (no penalties, no ideal term)."""
    X = xp.asarray(X, dtype=DTYPE)
    if X.ndim == 1:
        X = X[None, :]
    b, n = X.shape[0], X.shape[1] // 2
    cfg = _cfg(dataset)
    rows = _max_rows(n)
    if b > rows:
        return xp.concatenate([energy_production_batch(X[i:i + rows], dataset)
                               for i in range(0, b, rows)])
    return expected_power_batch(waked_speeds(X.reshape(b, n, 2), cfg["psi"]),
                                omega=cfg["omega"])


def min_spacing(dataset=1):
    return MIN_SPACING


def make_objective(farm_radius, chunk=None, dataset=1):
    """Convenience: bind the farm radius and data set for the optimizers."""
    def f(X):
        return objective_batch(X, farm_radius, chunk=chunk, dataset=dataset)
    return f
