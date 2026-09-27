"""
============================================================
HORNS REV 1 SITE DATA  (WFLOP_SITE=hornsrev)
============================================================

Real-site case: the Horns Rev 1 offshore wind farm (80 x Vestas V80).
All numbers below are copied verbatim from PyWake 2.6
(py_wake/examples/data/hornsrev1.py: wt_x, wt_y, power_curve, V80,
Hornsrev1Site), so the campaign does not need PyWake at run time.
validate_hornsrev.py re-imports PyWake and asserts that every array here
is identical to the PyWake source, then compares AEPs.

Only plain Python / NumPy is used here; objective_gpu.py moves the
arrays onto the active backend.

Geometry
--------
The feasible region is the convex hull of the 80 real turbine positions
(UTM zone 32N, metres), translated so that the hull's area centroid is at
the origin. The hull is computed with Andrew's monotone chain, so no
SciPy dependency; collinear points on the hull edges are dropped, leaving
the corner vertices only.
============================================================
"""

import numpy as np

# ---------------------------------------------------------------
# Turbine positions (PyWake hornsrev1.wt_x / wt_y), metres, UTM 32N
# ---------------------------------------------------------------
WT_X = [423974, 424042, 424111, 424179, 424247, 424315, 424384, 424452, 424534,
        424602, 424671, 424739, 424807, 424875, 424944, 425012, 425094, 425162,
        425231, 425299, 425367, 425435, 425504, 425572, 425654, 425722, 425791,
        425859, 425927, 425995, 426064, 426132, 426214, 426282, 426351, 426419,
        426487, 426555, 426624, 426692, 426774, 426842, 426911, 426979, 427047,
        427115, 427184, 427252, 427334, 427402, 427471, 427539, 427607, 427675,
        427744, 427812, 427894, 427962, 428031, 428099, 428167, 428235, 428304,
        428372, 428454, 428522, 428591, 428659, 428727, 428795, 428864, 428932,
        429014, 429082, 429151, 429219, 429287, 429355, 429424, 429492]
WT_Y = [6151447, 6150891, 6150335, 6149779, 6149224, 6148668, 6148112, 6147556,
        6151447, 6150891, 6150335, 6149779, 6149224, 6148668, 6148112, 6147556,
        6151447, 6150891, 6150335, 6149779, 6149224, 6148668, 6148112, 6147556,
        6151447, 6150891, 6150335, 6149779, 6149224, 6148668, 6148112, 6147556,
        6151447, 6150891, 6150335, 6149779, 6149224, 6148668, 6148112, 6147556,
        6151447, 6150891, 6150335, 6149779, 6149224, 6148668, 6148112, 6147556,
        6151447, 6150891, 6150335, 6149779, 6149224, 6148668, 6148112, 6147556,
        6151447, 6150891, 6150335, 6149779, 6149224, 6148668, 6148112, 6147556,
        6151447, 6150891, 6150335, 6149779, 6149224, 6148668, 6148112, 6147556,
        6151447, 6150891, 6150335, 6149779, 6149224, 6148668, 6148112, 6147556]
N_TURBINES = 80

# ---------------------------------------------------------------
# Vestas V80 (PyWake hornsrev1.power_curve), D = 80 m
# ---------------------------------------------------------------
DIAMETER = 80.0
ROTOR_RADIUS = 40.0
# (wind speed m/s, electrical power kW). PyWake interpolates linearly
# between the knots and returns 0 outside [3, 25] m/s.
POWER_CURVE_WS = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                  14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
                  24.0, 25.0]
POWER_CURVE_KW = [0.0, 66.6, 154.0, 282.0, 460.0, 696.0, 996.0, 1341.0, 1661.0,
                  1866.0, 1958.0, 1988.0, 1997.0, 1999.0, 2000.0, 2000.0, 2000.0,
                  2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0]
CUT_IN = POWER_CURVE_WS[0]           # 3 m/s (first knot, zero power)
CUT_OUT = POWER_CURVE_WS[-1]         # 25 m/s
# SIMPLIFICATION: the thrust coefficient is held constant at CT = 0.8 for
# every wind speed (the tabulated V80 ct curve, 0.82 at 4 m/s falling to
# 0.05 at 25 m/s, is NOT used). This keeps the per-pair wake deficit
# independent of the free-stream speed, which is what lets the expected
# power be written as a Weibull integral with a waked scale A*(1 - deficit),
# exactly as in the benchmark objective. It overstates wake losses above
# rated speed, where the real V80 ct is far below 0.8.
CT = 0.8

# ---------------------------------------------------------------
# Wind rose (PyWake Hornsrev1Site): 12 sectors of 30 deg, sector i centred
# on the meteorological direction 30*i deg (direction the wind comes FROM,
# clockwise from north). Frequencies are normalised to sum to 1, as PyWake
# does; A and k are the per-sector Weibull scale (m/s) and shape.
# ---------------------------------------------------------------
_F_RAW = [3.597152, 3.948682, 5.167395, 7.000154, 8.364547, 6.43485,
          8.643194, 11.77051, 15.15757, 14.73792, 10.01205, 5.165975]
WEIBULL_A = [9.176929, 9.782334, 9.531809, 9.909545, 10.04269, 9.593921,
             9.584007, 10.51499, 11.39895, 11.68746, 11.63732, 10.08803]
WEIBULL_K = [2.392578, 2.447266, 2.412109, 2.591797, 2.755859, 2.595703,
             2.583984, 2.548828, 2.470703, 2.607422, 2.626953, 2.326172]
SECTOR_FREQ = list(np.asarray(_F_RAW, dtype=np.float64) / np.sum(_F_RAW))
N_SECTORS = 12
SECTOR_WIDTH = 30.0
WD_MET = [SECTOR_WIDTH * i for i in range(N_SECTORS)]          # 0, 30, ... 330
TI = 0.1

# WFLOP_HR_DIRS: number of wind directions the objective evaluates (default 12,
# the sector centres). Twelve directions are too coarse for an AEP comparison:
# the as-built rows are aligned with sector centres and an optimizer can place
# turbines between the sampled directions. WFLOP_HR_DIRS=36 (or any multiple of
# 12) evaluates every 360/n degrees. Each direction takes the frequency, Weibull
# A and k of its nearest sector centre, which is what PyWake's Hornsrev1Site
# does for any direction grid. Frequencies are normalised to sum to 1.
HR_DIRS = int(__import__("os").environ.get("WFLOP_HR_DIRS", "12"))
if HR_DIRS % 12 or HR_DIRS < 12:
    raise ValueError(f"WFLOP_HR_DIRS must be a positive multiple of 12, not {HR_DIRS}")
if HR_DIRS != 12:
    _wd = np.arange(HR_DIRS) * (360.0 / HR_DIRS)
    _sec = np.rint(_wd / SECTOR_WIDTH).astype(int) % 12      # nearest sector centre
    _per = lambda v: np.asarray(v, dtype=np.float64)[_sec]
    _f = _per(_F_RAW)
    SECTOR_FREQ = list(_f / _f.sum())
    WEIBULL_A = list(_per(WEIBULL_A))
    WEIBULL_K = list(_per(WEIBULL_K))
    N_SECTORS = HR_DIRS
    SECTOR_WIDTH = 360.0 / HR_DIRS
    WD_MET = list(_wd)

# Direction convention. The objective's wake geometry uses THETA, the
# MATHEMATICAL angle (anticlockwise from +x = east) of the direction the wind
# blows TOWARDS. A meteorological direction wd (wind FROM wd, clockwise from
# north) blows towards the compass bearing wd + 180, i.e. the vector
# (-sin wd, -cos wd), whose mathematical angle is 270 - wd.
THETA_DEG = [(270.0 - wd) % 360.0 for wd in WD_MET]

# ---------------------------------------------------------------
# Speed bins for the expected-power integral: 0.25 m/s from cut-in to
# cut-out, midpoint power (linear interpolation of the tabulated curve)
# times the Weibull probability of the bin.
# ---------------------------------------------------------------
SPEED_BIN = 0.25
SPEED_EDGES = [CUT_IN + SPEED_BIN * i
               for i in range(int(round((CUT_OUT - CUT_IN) / SPEED_BIN)) + 1)]
SPEED_MIDS = [0.5 * (a + b) for a, b in zip(SPEED_EDGES[:-1], SPEED_EDGES[1:])]
POWER_MID_KW = list(np.interp(SPEED_MIDS, POWER_CURVE_WS, POWER_CURVE_KW))


# ---------------------------------------------------------------
# Boundary polygon
# ---------------------------------------------------------------
def _convex_hull(pts):
    """Andrew's monotone chain; counter-clockwise, collinear points dropped."""
    p = sorted(set(map(tuple, pts)))
    if len(p) < 3:
        return np.asarray(p, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _polygon_centroid(V):
    # Work relative to the first vertex: with raw UTM northings (~6e6 m) the
    # shoelace products lose ~1e-8 m of precision in the centroid.
    ref = V[0]
    V = V - ref
    x, y = V[:, 0], V[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    c = x * ys - xs * y
    area = 0.5 * np.sum(c)
    cx = np.sum((x + xs) * c) / (6.0 * area)
    cy = np.sum((y + ys) * c) / (6.0 * area)
    return area, np.array([cx, cy]) + ref


_XY_UTM = np.stack([np.asarray(WT_X, dtype=np.float64),
                    np.asarray(WT_Y, dtype=np.float64)], axis=1)
_HULL_UTM = _convex_hull([(int(a), int(b)) for a, b in zip(WT_X, WT_Y)])
POLYGON_AREA, CENTROID_UTM = _polygon_centroid(_HULL_UTM)

# Everything below is in the translated frame (hull centroid at the origin).
POLYGON = _HULL_UTM - CENTROID_UTM             # (V, 2) counter-clockwise
REAL_LAYOUT = _XY_UTM - CENTROID_UTM           # (80, 2) the as-built layout

# Optimizer search box: [-h, h]^2 with h = half the larger side of the
# polygon's bounding box. The hull is (to the metre) a parallelogram and so
# point-symmetric about its centroid; the assertion below proves the box
# contains the whole polygon, so no feasible layout is cut off by the box.
_BB_LO, _BB_HI = POLYGON.min(axis=0), POLYGON.max(axis=0)
BOX_HALF = 0.5 * float(np.max(_BB_HI - _BB_LO))
assert np.all(np.abs(POLYGON) <= BOX_HALF + 1e-9), \
    "search box [-h, h]^2 does not contain the Horns Rev polygon"

# Distances within this tolerance outside the polygon are treated as on the
# boundary. It absorbs floating-point round-off (~1e-12 m) of the polygon
# projection used by the GNN repair and of points that lie exactly on a hull
# edge (the real turbines of the outer rows), so neither picks up a spurious
# penalty of (1 + 1e10 * 1e-12)^2. One micrometre is physically irrelevant.
BOUNDARY_TOL = 1e-6


def summary():
    return {
        "turbines": N_TURBINES, "diameter_m": DIAMETER, "ct": CT, "ti": TI,
        "polygon_vertices_m": POLYGON.tolist(),
        "polygon_area_km2": POLYGON_AREA / 1e6,
        "centroid_utm32n_m": CENTROID_UTM.tolist(),
        "box_half_m": BOX_HALF,
        "sector_freq": SECTOR_FREQ, "weibull_A": WEIBULL_A,
        "weibull_k": WEIBULL_K, "wd_met_deg": WD_MET, "theta_deg": THETA_DEG,
        "n_directions": N_SECTORS,
        "speed_bin_ms": SPEED_BIN, "cut_in": CUT_IN, "cut_out": CUT_OUT,
        "min_spacing_m": 4 * DIAMETER,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=1))
    nn = np.sqrt(((REAL_LAYOUT[:, None] - REAL_LAYOUT[None]) ** 2).sum(-1))
    nn[np.arange(80), np.arange(80)] = np.inf
    print("real layout min spacing (m):", nn.min(), " = %.2f D" % (nn.min() / DIAMETER))
    print("bounding box (m):", _BB_LO, _BB_HI)
