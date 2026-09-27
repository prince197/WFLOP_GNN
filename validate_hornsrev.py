"""
============================================================
HORNS REV 1 VALIDATION AGAINST PyWake   (WFLOP_SITE=hornsrev)
============================================================

    WFLOP_BACKEND=cpu python validate_hornsrev.py

Needs PyWake (pip install py_wake); the campaign itself does not.

1. DATA. Every array in site_hornsrev.py (turbine positions, V80 power
   curve, 12-sector frequencies / Weibull A / k, TI) is compared with the
   PyWake source it was copied from.
2. FEASIBILITY of the as-built layout under the campaign constraints
   (inside the hull polygon, pairwise spacing >= 4D, zero penalty).
3. AEP of the as-built layout from objective_gpu.py and from PyWake
   configured identically:
       V80 power curve (kW), constant CT = 0.8 (PowerCtTabular with a flat
       ct curve), Hornsrev1Site wind rose (TI 0.1), PropagateDownwind,
       SquaredSum superposition, RotorCenter (no rotor averaging), no
       turbulence model, no blockage, ct2a = ct2a_mom1d (a = (1-sqrt(1-CT))/2,
       the relation the objective uses; PyWake's default ct2a_madsen differs)
         jensen   : NOJDeficit(k = 0.04)
         gaussian : BastankhahGaussianDeficit(k = 0.3837*0.1 + 0.003678
                    = 0.04205, ceps = 0.2, use_effective_ws = False)
   Two PyWake discretisations are reported:
     A  like-for-like: the 12 sector-centre directions and the objective's
        0.25 m/s speed bins (edges 3.0 .. 25.0). The only remaining
        difference is where the speed bins sit: the objective integrates the
        power curve against the WAKED Weibull distribution (bins in the
        turbine's own speed), PyWake bins the FREE-STREAM speed and evaluates
        power at u_mid*(1 - deficit). This is the acceptance test (< 0.5 %).
        Measured (PyWake 2.6.20): +0.053 % Jensen, +0.040 % Gaussian. Almost
        all of it is free-stream speeds ABOVE 25 m/s: PyWake's speed grid
        stops at 25 m/s, while the waked-Weibull integral still credits a
        waked turbine whose own speed is below cut-out when the free stream
        is above it (its upstream neighbour keeping CT = 0.8). Extending
        PyWake's grid to 45 m/s with ct = 0.8 everywhere and 0.05 m/s bins
        gives 635.5527 / 671.4101 GWh against the objective's 635.5521 /
        671.4097 GWh at 0.01 m/s bins: the two agree to 1e-6.
     B  PyWake's own defaults: 1-degree directions (sector data by nearest
        sector) and 1 m/s speeds 3..25. Wakes are then smeared over each 30
        degree sector instead of being evaluated at its centre only, so for
        a regular layout whose rows line up with sector centres the wake loss
        differs materially (measured: the objective's AEP is 1.96 % (Jensen)
        and 3.01 % (Gaussian) BELOW PyWake's 1-degree AEP; the as-built rows
        are aligned with the 270 deg sector centre). Reported for context,
        not as a pass/fail test: evaluating each sector at its centre
        direction only is the benchmark objective's convention.
============================================================
"""

import os
import sys

os.environ.setdefault("WFLOP_BACKEND", "cpu")
os.environ["WFLOP_SITE"] = "hornsrev"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np

TOL_PCT = 0.5


def load_objective(wake):
    os.environ["WFLOP_WAKE"] = wake
    for m in ("objective_gpu", "site_hornsrev", "backend"):
        sys.modules.pop(m, None)
    import objective_gpu as og
    assert og.SITE == "hornsrev" and og.WAKE_MODEL == wake
    return og


def check_data():
    import site_hornsrev as S
    from py_wake.examples.data import hornsrev1 as H
    site = H.Hornsrev1Site()
    # XRSite appends a wrap-around copy of sector 0 at 360 deg; drop it
    ds = site.ds.sel(wd=site.ds.wd.values[site.ds.wd.values < 360])
    checks = {
        "wt_x": np.array_equal(np.asarray(S.WT_X, float), np.asarray(H.wt_x, float)),
        "wt_y": np.array_equal(np.asarray(S.WT_Y, float), np.asarray(H.wt_y, float)),
        "power_curve_ws": np.array_equal(S.POWER_CURVE_WS, H.power_curve[:, 0]),
        "power_curve_kW": np.allclose(np.asarray(S.POWER_CURVE_KW) * 1000.0,
                                      H.power_curve[:, 1], rtol=0, atol=1e-9),
        "sector_freq": np.allclose(S.SECTOR_FREQ, ds.Sector_frequency.values,
                                   rtol=0, atol=1e-15),
        "weibull_A": np.array_equal(S.WEIBULL_A, ds.Weibull_A.values),
        "weibull_k": np.array_equal(S.WEIBULL_K, ds.Weibull_k.values),
        "sector_centres": np.array_equal(S.WD_MET, ds.wd.values),
        "ti": float(ds.TI.values) == S.TI,
        "diameter": H.V80().diameter() == S.DIAMETER,
    }
    for k, ok in checks.items():
        print(f"  {k:16s} {'identical' if ok else 'MISMATCH'}")
    return all(checks.values())


def check_feasibility(og):
    import site_hornsrev as S
    xy = S.REAL_LAYOUT
    P = og.xp.asarray(xy[None], dtype=og.DTYPE)
    dist = np.asarray(og.polygon_nearest(P)[0])[0]
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    d[np.arange(len(xy)), np.arange(len(xy))] = np.inf
    bpen = float(og.polygon_penalty_batch(P)[0])
    spen = float(og.spacing_penalty_batch(P)[0])
    on_edge = int(np.sum(np.asarray(og.polygon_nearest(P)[2])[0]))
    print(f"  polygon: {len(S.POLYGON)} vertices, area {S.POLYGON_AREA / 1e6:.3f} km^2, "
          f"search box [-{S.BOX_HALF:.1f}, {S.BOX_HALF:.1f}] m")
    print(f"  as-built layout: max distance outside polygon {dist.max():.3g} m "
          f"({on_edge} turbines numerically a hair outside an edge, all within "
          f"the {S.BOUNDARY_TOL:g} m tolerance)")
    print(f"  min pairwise spacing {d.min():.1f} m = {d.min() / S.DIAMETER:.2f} D "
          f"(limit {og.MIN_SPACING:.0f} m = 4D)")
    print(f"  boundary penalty {bpen}, spacing penalty {spen}")
    return dist.max() <= S.BOUNDARY_TOL and d.min() >= og.MIN_SPACING \
        and bpen == 0.0 and spen == 0.0


def pywake_aep(wake, og, like_for_like):
    import site_hornsrev as S
    from py_wake.examples.data.hornsrev1 import Hornsrev1Site, wt_x, wt_y
    from py_wake.wind_turbines import WindTurbine
    from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
    from py_wake.wind_farm_models import PropagateDownwind
    from py_wake.deficit_models.noj import NOJDeficit
    from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit
    from py_wake.deficit_models.utils import ct2a_mom1d
    from py_wake.superposition_models import SquaredSum
    from py_wake.rotor_avg_models import RotorCenter

    site = Hornsrev1Site(ti=S.TI)
    ws_tab = np.asarray(S.POWER_CURVE_WS)
    wt = WindTurbine(name="V80_ct0.8", diameter=S.DIAMETER, hub_height=70.0,
                     powerCtFunction=PowerCtTabular(
                         ws_tab, np.asarray(S.POWER_CURVE_KW), "kW",
                         np.full(ws_tab.shape, S.CT), method="linear"))
    if wake == "jensen":
        dm = NOJDeficit(k=og.K, ct2a=ct2a_mom1d, rotorAvgModel=RotorCenter())
    else:
        dm = BastankhahGaussianDeficit(k=og.K_STAR, ceps=0.2, ct2a=ct2a_mom1d,
                                       use_effective_ws=False,
                                       rotorAvgModel=RotorCenter())
    wfm = PropagateDownwind(site, wt, wake_deficitModel=dm,
                            superpositionModel=SquaredSum(),
                            deflectionModel=None, turbulenceModel=None)
    if like_for_like:
        sim = wfm(wt_x, wt_y, wd=np.asarray(S.WD_MET, float),
                  ws=np.asarray(S.SPEED_MIDS, float))
    else:
        sim = wfm(wt_x, wt_y)          # PyWake defaults: wd 0..359, ws 3..25
    aep = float(sim.aep().sum())
    aep_free = float(sim.aep(with_wake_loss=False).sum())
    return aep, aep_free


def main():
    ok = True
    print("=" * 74)
    print("1. site data vs PyWake source")
    load_objective("jensen")
    ok &= check_data()

    print("2. feasibility of the as-built Horns Rev 1 layout")
    og = load_objective("jensen")
    ok &= check_feasibility(og)

    print("3. AEP of the as-built layout (GWh/yr)")
    rows = []
    for wake in ("jensen", "gaussian"):
        og = load_objective(wake)
        import site_hornsrev as S
        X = S.REAL_LAYOUT.reshape(1, -1)
        ep = float(og.energy_production_batch(X)[0])
        obj = float(og.objective_batch(X, 0)[0])
        aep = og.aep_gwh(ep)
        aep_ideal = og.aep_gwh(og.HR_IDEAL * S.N_TURBINES)
        assert abs(obj - (og.HR_IDEAL * S.N_TURBINES - ep)) < 1e-6 * obj
        pa, pa_free = pywake_aep(wake, og, like_for_like=True)
        pb, pb_free = pywake_aep(wake, og, like_for_like=False)
        rel_a = 100.0 * (aep - pa) / pa
        rel_b = 100.0 * (aep - pb) / pb
        rows.append((wake, aep, pa, rel_a, pb, rel_b))
        print(f"  {wake:8s} objective     AEP {aep:9.3f}   wake-free {aep_ideal:9.3f}"
              f"   wake loss {100 * (1 - aep / aep_ideal):6.3f} %"
              f"   (objective value {obj:.3f} kW)")
        print(f"  {'':8s} PyWake A     AEP {pa:9.3f}   wake-free {pa_free:9.3f}"
              f"   wake loss {100 * (1 - pa / pa_free):6.3f} %   rel. diff {rel_a:+.4f} %")
        print(f"  {'':8s} PyWake B     AEP {pb:9.3f}   wake-free {pb_free:9.3f}"
              f"   wake loss {100 * (1 - pb / pb_free):6.3f} %   rel. diff {rel_b:+.4f} %"
              "   (1-deg directions, context only)")
        ok &= abs(rel_a) < TOL_PCT
    print("=" * 74)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
