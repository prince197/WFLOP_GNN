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

SITES
-----
WFLOP_SITE=benchmark (default) is the circular-farm benchmark above,
unchanged bit for bit. WFLOP_SITE=hornsrev switches to the real Horns
Rev 1 case (Vestas V80, 12-sector wind rose, convex-hull boundary); see
the HORNS REV 1 section at the end of this file and site_hornsrev.py.
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
SITE = __import__("os").environ.get("WFLOP_SITE", "benchmark").lower()
if SITE not in ("benchmark", "hornsrev"):
    raise ValueError(f"WFLOP_SITE must be 'benchmark' or 'hornsrev', not {SITE!r}")

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
    if SITE == "hornsrev":
        return _hr_waked_scale(P, psi)
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
    if SITE == "hornsrev":
        # The site fixes the wind rose; the benchmark data-set number is
        # ignored (the runner keeps it at 1 only for file naming).
        return HR_CFG
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
    if SITE == "hornsrev":
        return _hr_objective_batch(X)
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
    if SITE == "hornsrev":
        return _hr_energy_batch(X)
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


def search_box(farm_radius):
    """(lb, ub) of the square search box the optimizers work in.

    benchmark : [-radius, radius], the box around the circular farm.
    hornsrev  : [-h, h], h = half the larger side of the polygon's bounding
                box (farm_radius is ignored; the runner stores it as 0).
    """
    if SITE == "hornsrev":
        return -HR_BOX_HALF, HR_BOX_HALF
    return -farm_radius, farm_radius


# ===============================================================
# HORNS REV 1   (WFLOP_SITE=hornsrev)
# ===============================================================
# Real-site case. Everything above is left exactly as it was; this block only
# rebinds the module constants the other modules read (R, K, CT, MIN_SPACING,
# TI, K_STAR, ...) and adds the site's own wake, power and boundary functions,
# which objective_batch / energy_production_batch / waked_speeds / _cfg
# dispatch to when SITE == "hornsrev".
#
# Turbine  Vestas V80, D = 80 m (R = 40 m), PyWake V80 power curve in kW.
#          SIMPLIFICATION: constant thrust coefficient CT = 0.8 at every wind
#          speed (the tabulated V80 ct curve is not used), so the per-pair
#          deficit does not depend on the free-stream speed.
# Wind     PyWake Hornsrev1Site: 12 sectors x (frequency, Weibull A, k),
#          TI = 0.1. Expected power per turbine
#              sum_s f_s * integral P(v) Weibull(v; A_s (1 - d_is), k_s) dv
#          i.e. the free-stream Weibull scale is reduced by the turbine's
#          waked deficit d_is, integrated over 0.25 m/s bins from cut-in (3)
#          to cut-out (25 m/s): midpoint power times bin probability. kW; no
#          sector-width factor (the frequencies sum to 1).
# Wakes    Jensen (NOJ) top-hat, K = 0.04 (standard offshore value), or the
#          Bastankhah & Porte-Agel Gaussian with k* = 0.3837 TI + 0.003678,
#          TI = 0.1 -> k* = 0.04205. Root-sum-square superposition and a
#          0.999 cap on the combined deficit for both.
#          DIFFERENCE FROM THE BENCHMARK JENSEN: the benchmark's cone test
#          (angle from a virtual apex R/K upstream of the source) has no
#          downstream check, so a turbine up to R/K upstream of a source and
#          inside the reversed cone also receives its deficit. With K = 0.04
#          R/K = 1000 m, and the Horns Rev rows are aligned with the 270 deg
#          sector, so that leak would be large here. The Horns Rev Jensen
#          therefore uses the standard NOJ condition: waked only if DOWNSTREAM
#          (x > 0) and within the wake radius R + K x of the source axis -
#          the same cone, restricted to x > 0. This matches PyWake's
#          NOJDeficit to round-off. The benchmark path is NOT changed.
# Objective (kW, minimised)
#          ideal wake-free farm power - expected farm power + penalties
# Boundary convex hull of the 80 as-built turbines, centred on its area
#          centroid. Penalty (1 + 1e10 d)^2 per turbine at distance d > 0
#          (1 um tolerance, see site_hornsrev.BOUNDARY_TOL) outside it;
#          spacing penalty exactly as the benchmark with 4D = 320 m.
if SITE == "hornsrev":
    import site_hornsrev as HR

    R = HR.ROTOR_RADIUS                      # 40 m
    K = 0.04                                 # offshore Jensen wake decay
    CT = HR.CT                               # 0.8, constant (simplification)
    A = 1 - (1 - CT) ** 0.5
    B_COEF = K / R
    ALPHA = float(xp.arctan(xp.asarray(K, dtype=xp.float64)))
    MIN_SPACING = 8 * R                      # 4D = 320 m, same rule as benchmark
    D_ROTOR = 2.0 * R
    TI = float(__import__("os").environ.get("WFLOP_TI", HR.TI))
    K_STAR = 0.3837 * TI + 0.003678
    _BETA_G = 0.5 * (1.0 + (1.0 - CT) ** 0.5) / (1.0 - CT) ** 0.5
    EPS_G = 0.2 * _BETA_G ** 0.5

    HR_N_SECTORS = HR.N_SECTORS
    HR_THETA = xp.asarray(HR.THETA_DEG, dtype=DTYPE) * (math.pi / 180.0)  # (12,)
    HR_COS = xp.cos(HR_THETA)[None, :, None, None]                        # (1,12,1,1)
    HR_SIN = xp.sin(HR_THETA)[None, :, None, None]
    HR_FREQ = xp.asarray(HR.SECTOR_FREQ, dtype=DTYPE)                     # (12,)
    HR_A = xp.asarray(HR.WEIBULL_A, dtype=DTYPE)                          # (12,)
    HR_KSHAPE = xp.asarray(HR.WEIBULL_K, dtype=DTYPE)                     # (12,)

    # Expected power by summation by parts: with bin edges e_0..e_M, midpoint
    # powers p_1..p_M and G(e) = exp(-(e/a)^k) (Weibull survival),
    #     sum_m p_m [G(e_{m-1}) - G(e_m)] = sum_j w_j G(e_j),
    #     w_0 = p_1,  w_j = p_{j+1} - p_j,  w_M = -p_M.
    # (e_j/a)^k = e_j^k * a^-k, and e_j^k depends only on the sector's shape
    # k, so it is tabulated once: one exp per (edge, sector, turbine, layout).
    import numpy as _np
    _pm = _np.asarray(HR.POWER_MID_KW, dtype=_np.float64)
    _w = _np.concatenate([[_pm[0]], _np.diff(_pm), [-_pm[-1]]])
    _e = _np.asarray(HR.SPEED_EDGES, dtype=_np.float64)
    HR_EDGE_W = [float(v) for v in _w]                                    # (M+1,)
    HR_EDGE_POW = xp.asarray(_e[:, None] ** _np.asarray(HR.WEIBULL_K)[None, :],
                             dtype=DTYPE)                                 # (M+1,12)

    HR_POLY = xp.asarray(HR.POLYGON, dtype=DTYPE)                         # (V,2) ccw
    _HR_EDGE = xp.asarray(_np.roll(HR.POLYGON, -1, axis=0) - HR.POLYGON,
                          dtype=DTYPE)                                    # (V,2)
    _HR_EDGE_L2 = xp.sum(_HR_EDGE * _HR_EDGE, axis=-1)                    # (V,)
    HR_BOX_HALF = HR.BOX_HALF
    HR_BOUNDARY_TOL = HR.BOUNDARY_TOL


def _hr_total_deficit(P):
    """P : (B,N,2) -> (B,12,N) combined fractional deficit per sector."""
    x = P[:, None, :, 0]
    y = P[:, None, :, 1]
    dx = x[..., :, None] - x[..., None, :]          # (B,1,N,N) : i - j
    dy = y[..., :, None] - y[..., None, :]
    proj = dx * HR_COS + dy * HR_SIN                # (B,12,N,N) downstream distance of i from j
    cw = dy * HR_COS - dx * HR_SIN                  # crosswind offset
    lat2 = cw * cw
    n = P.shape[1]
    eye = xp.eye(n, dtype=bool)[None, None, :, :]
    downstream = (proj > 0.0) & (~eye)
    if WAKE_MODEL == "gaussian":
        deficit = _gaussian_deficit(proj, lat2)
        waked = downstream
    else:
        # NOJ: inside the top-hat wake of radius R + K x, downstream only.
        # (For x > 0 this is exactly the benchmark cone test beta < arctan K.)
        wr = R + K * proj
        waked = downstream & (lat2 < wr * wr)
        deficit = A / ((1.0 + B_COEF * xp.maximum(proj, 0.0)) ** 2)
    total = xp.sqrt(xp.sum(xp.where(waked, deficit * deficit, 0.0), axis=-1))
    return xp.minimum(total, 0.999)


def _hr_waked_scale(P, psi=None):
    """(B,N,2) -> (B,12,N) waked Weibull scale A_s (1 - d_is)."""
    base = HR_A if psi is None else psi
    return base[None, :, None] * (1.0 - _hr_total_deficit(P))


def _hr_expected_power(scale):
    """scale : (B,12,N) Weibull scale seen by each turbine -> (B,) expected
    farm power in kW."""
    ak = scale ** (-HR_KSHAPE[None, :, None])                 # a^-k
    acc = xp.zeros_like(scale)
    for j, wj in enumerate(HR_EDGE_W):
        if wj != 0.0:
            acc = acc + wj * xp.exp(-(HR_EDGE_POW[j][None, :, None] * ak))
    per_turbine = xp.sum(HR_FREQ[None, :, None] * acc, axis=1)  # (B,N)
    return xp.sum(per_turbine, axis=1)


def _hr_rows(n):
    item = xp.dtype(DTYPE).itemsize
    per_row = HR_N_SECTORS * n * n * item * 8
    return max(1, int(MEM_BUDGET_BYTES // max(per_row, 1)))


def polygon_nearest(P):
    """P : (...,N,2) -> (dist, nearest, outside)

    dist     (...,N) Euclidean distance to the polygon (0 inside/on it)
    nearest  (...,N,2) nearest point of the polygon (P itself when inside)
    outside  (...,N) bool, strictly outside some edge's half-plane
    Vectorised over every leading axis; the polygon is convex and ccw.
    """
    rel = P[..., None, :] - HR_POLY                           # (...,N,V,2)
    cross = _HR_EDGE[:, 0] * rel[..., 1] - _HR_EDGE[:, 1] * rel[..., 0]
    outside = xp.any(cross < 0.0, axis=-1)
    t = xp.clip(xp.sum(rel * _HR_EDGE, axis=-1) / _HR_EDGE_L2, 0.0, 1.0)
    q = HR_POLY + t[..., None] * _HR_EDGE                     # (...,N,V,2)
    d2 = xp.sum((P[..., None, :] - q) ** 2, axis=-1)          # (...,N,V)
    j = xp.argmin(d2, axis=-1)
    qn = xp.take_along_axis(q, j[..., None, None], axis=-2)[..., 0, :]
    dist = xp.where(outside, xp.sqrt(xp.min(d2, axis=-1)), 0.0)
    nearest = xp.where(outside[..., None], qn, P)
    return dist, nearest, outside


def project_to_polygon(P):
    """Move every point outside the polygon to the nearest point on it."""
    return polygon_nearest(P)[1]


def polygon_penalty_batch(P):
    """(B,N,2) -> (B,): (1 + 1e10 d)^2 per turbine outside the polygon."""
    d = polygon_nearest(P)[0]
    viol = d > HR_BOUNDARY_TOL
    term = xp.where(viol, (1.0 + PENALTY * d) ** 2, 0.0)
    return xp.sum(term, axis=1)


def _hr_objective_batch(X):
    X = xp.asarray(X, dtype=DTYPE)
    if X.ndim == 1:
        X = X[None, :]
    b, n = X.shape[0], X.shape[1] // 2
    rows = _hr_rows(n)
    if b > rows:
        return xp.concatenate([_hr_objective_batch(X[i:i + rows])
                               for i in range(0, b, rows)])
    P = X.reshape(b, n, 2)
    ep = _hr_expected_power(_hr_waked_scale(P))
    loss = HR_IDEAL * n - ep
    loss = loss + polygon_penalty_batch(P)
    loss = loss + spacing_penalty_batch(P)
    return loss


def _hr_energy_batch(X):
    X = xp.asarray(X, dtype=DTYPE)
    if X.ndim == 1:
        X = X[None, :]
    b, n = X.shape[0], X.shape[1] // 2
    rows = _hr_rows(n)
    if b > rows:
        return xp.concatenate([_hr_energy_batch(X[i:i + rows])
                               for i in range(0, b, rows)])
    return _hr_expected_power(_hr_waked_scale(X.reshape(b, n, 2)))


def aep_gwh(expected_power_kw):
    """Expected farm power (kW) -> annual energy production (GWh/yr)."""
    return expected_power_kw * 8760.0 / 1e6


if SITE == "hornsrev":
    # Ideal (wake-free) expected power of one V80 under the Horns Rev rose, kW.
    HR_IDEAL = float(_hr_expected_power(
        xp.broadcast_to(HR_A[None, :, None], (1, HR_N_SECTORS, 1)).copy())[0])
    HR_CFG = dict(omega=HR_FREQ, psi=HR_A, ideal=HR_IDEAL)
