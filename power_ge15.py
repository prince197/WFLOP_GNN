"""GE 1.5 MW, 77 m rotor power curve (WFLOP_POWER=ge15).

Source: NREL turbine-models library, Onshore/DOE_GE_1.5MW_77.csv
(https://github.com/NREL/turbine-models, BSD-3-Clause), the DOE-owned
GE 1.5 MW turbine at NREL's Flatirons Campus: rated 1,500 kW, rotor
diameter 77 m, cut-in 3.5 m/s, rated 14.5 m/s, cut-out 25 m/s.

The benchmark turbine of this package has the same rotor (R = 38.5 m),
rating, cut-in and cut-out, with a linearised ramp to 14 m/s. Only the
power curve changes under WFLOP_POWER=ge15; the thrust coefficient stays
at the benchmark's constant CT = 0.8.

Processing: the table rows below 3.5 m/s (standby consumption) are set to
zero, the values are clipped to [0, 1500] kW, and the curve is held at
1,500 kW from the last tabulated speed (21.45 m/s) to cut-out (25 m/s).
Above 25 m/s the output is zero.
"""
import numpy as np

CUT_IN = 3.5
CUT_OUT = 25.0
P_RATED = 1500.0

_TABLE = [  # wind speed (m/s), power (kW), as published
    (1.01, -4.92),
    (1.53, -5.44),
    (1.99, -5.78),
    (2.43, -5.56),
    (2.97, 0.59),
    (3.51, 18.91),
    (3.97, 58.88),
    (4.52, 100.82),
    (5.01, 162.89),
    (5.52, 235.1),
    (5.99, 303.63),
    (6.52, 399.01),
    (7.04, 513.9),
    (7.51, 608.8),
    (8, 742.33),
    (8.49, 853.63),
    (9.01, 975.43),
    (9.52, 1096.64),
    (10.03, 1200),
    (10.48, 1260),
    (10.99, 1318),
    (11.49, 1390),
    (11.98, 1400),
    (12.52, 1453),
    (12.97, 1452),
    (13.51, 1478),
    (13.97, 1482),
    (14.43, 1496),
    (15.01, 1498),
    (15.49, 1495),
    (15.98, 1497),
    (16.46, 1505),
    (16.97, 1506),
    (17.52, 1512),
    (18, 1497),
    (18.47, 1509),
    (19.06, 1499),
    (19.4, 1502),
    (19.96, 1511),
    (20.51, 1508),
    (20.95, 1503),
    (21.45, 1499),
]

WS = np.array([r[0] for r in _TABLE] + [CUT_OUT], dtype=np.float64)
PW = np.clip(np.array([r[1] for r in _TABLE] + [P_RATED], dtype=np.float64), 0.0, P_RATED)


def power(s):
    """Power (kW) at hub-height speed s, zero outside [cut-in, cut-out]."""
    s = np.asarray(s, dtype=np.float64)
    return np.where((s >= CUT_IN) & (s <= CUT_OUT), np.interp(s, WS, PW), 0.0)


BIN = 0.25                                           # m/s, midpoint rule
EDGES = np.arange(CUT_IN, CUT_OUT + 1e-9, BIN)        # 3.5 ... 25.0  (87 edges)
MIDS = 0.5 * (EDGES[:-1] + EDGES[1:])
P_MID = power(MIDS)                                   # (86,)
