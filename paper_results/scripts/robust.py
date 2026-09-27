"""
Objective- and rose-robustness analysis, written to robust.pkl for report.py.

Two scorings of the SAME stored layouts:
    A  linear curve  (the campaign objective; validation)
    B  cubic curve   (power-curve robustness)

Nothing is re-optimized. These are transfer tests: they ask whether the
ordering the campaign reports is a property of the layouts or of the
scoring function used to produce them.
"""
import pickle
import sys
from collections import Counter

import numpy as np
from scipy.stats import rankdata, wilcoxon, friedmanchisquare

sys.path.insert(0, "/home/claude/wflop")
from rescore import load, DS, expected_power, f_linear, f_cubic

ALGS = ["GA", "PSO", "DE", "GWO", "BBO", "SSA", "LXSSA", "QASSA", "PF",
        "GNNLXSSA"]
NAME = {"GNNLXSSA": "GNN-LX-SSA", "LXSSA": "LX-SSA", "QASSA": "QA-SSA",
        "GA": "EA (GA)"}
REF = "GNNLXSSA"


def ideal(fn, *a):
    return fn(np.zeros((1, 1, 2)), *a)[0]


def per_config(rows, loss, feas):
    cfg = {}
    for i, rw in enumerate(rows):
        if rw["alg"] not in ALGS:
            continue
        cfg.setdefault((rw["radius"], rw["n"]), {})[rw["alg"]] = (loss[i],
                                                                  bool(feas[i]))
    return cfg


def ranks(cfg):
    """-> {alg: [rank per configuration]}, list of winners, n compared."""
    acc, win, n = {}, [], 0
    for key in sorted(cfg):
        d = cfg[key]
        alg = [a for a in ALGS if a in d and d[a][1]]
        if len(alg) < 3:
            continue
        L = np.array([d[a][0] for a in alg])
        if np.ptp(L) < 1e-9:            # every algorithm found the same layout
            continue
        r = rankdata(L)
        for a, v in zip(alg, r):
            acc.setdefault(a, []).append(v)
        win.append(alg[int(np.argmin(L))])
        n += 1
    return acc, win, n


def paired(cfgA, cfgB):
    """Rank agreement between two scorings, over common configurations."""
    changed = tot = 0
    for key in sorted(cfgA):
        dA, dB = cfgA[key], cfgB.get(key, {})
        alg = [a for a in ALGS if a in dA and dA[a][1] and a in dB]
        if len(alg) < 3:
            continue
        LA = np.array([dA[a][0] for a in alg])
        LB = np.array([dB[a][0] for a in alg])
        if np.ptp(LA) < 1e-9:
            continue
        tot += 1
        if not np.array_equal(rankdata(LA), rankdata(LB)):
            changed += 1
    return changed, tot


def wilcox_vs_ref(cfg):
    """GNN-LX-SSA against each competitor on paired per-configuration loss."""
    out = {}
    for a in ALGS:
        if a == REF:
            continue
        x, y = [], []
        for key in sorted(cfg):
            d = cfg[key]
            if REF in d and a in d and d[REF][1] and d[a][1]:
                x.append(d[REF][0])
                y.append(d[a][0])
        x, y = np.array(x), np.array(y)
        if len(x) < 6:
            continue
        dif = y - x                      # positive => competitor worse
        nz = dif != 0
        if nz.sum() < 6:
            continue
        st, p = wilcoxon(x[nz], y[nz])
        rk = rankdata(np.abs(dif[nz]))
        Rp = rk[dif[nz] > 0].sum()
        Rm = rk[dif[nz] < 0].sum()
        r = (Rp - Rm) / (Rp + Rm)
        out[a] = dict(n=int(nz.sum()), Rplus=float(Rp), Rminus=float(Rm),
                      p=float(p), r=float(r),
                      wins=int((dif > 0).sum()), losses=int((dif < 0).sum()))
    return out


def friedman(acc):
    algs = [a for a in ALGS if a in acc]
    L = min(len(acc[a]) for a in algs)
    arrs = [np.array(acc[a][:L]) for a in algs]
    st, p = friedmanchisquare(*arrs)
    return float(st), float(p), L


def main():
    res = {}

    for ds, path in DS.items():
        rows = load(path)
        by_n = {}
        for i, rw in enumerate(rows):
            by_n.setdefault(rw["n"], []).append(i)
        feas = np.array([rw["wake"] < 1e6 for rw in rows])
        nrm = len(rows)
        lin = np.zeros(nrm); cub = np.zeros(nrm)
        for n, idx in sorted(by_n.items()):
            P = np.stack([rows[i]["P"] for i in idx])
            lin[idx] = expected_power(P, ds, f_linear)
            cub[idx] = expected_power(P, ds, f_cubic)

        il = ideal(expected_power, ds, f_linear)
        ic = ideal(expected_power, ds, f_cubic)
        nt = np.array([rw["n"] for rw in rows], dtype=float)

        scor = {"linear": il * nt - lin, "cubic": ic * nt - cub}
        d = {"ideal_linear": il, "ideal_cubic": ic,
             "overstatement": il / ic}
        cfgs = {k: per_config(rows, v, feas) for k, v in scor.items()}
        for k, cfg in cfgs.items():
            acc, win, n = ranks(cfg)
            st, p, L = friedman(acc)
            d[k] = dict(meanrank={a: float(np.mean(v)) for a, v in acc.items()},
                        winners=Counter(win).most_common(), n=n,
                        friedman=(st, p, L), wilcoxon=wilcox_vs_ref(cfg))
        d["rank_change_cubic"] = paired(cfgs["linear"], cfgs["cubic"])
        res[ds] = d

    with open("/home/claude/wflop/robust.pkl", "wb") as fh:
        pickle.dump(res, fh)

    for ds in (1, 2):
        d = res[ds]
        print(f"\n===== WIND DATA SET {ds} "
              f"(linear overstates ideal by {100*(d['overstatement']-1):.1f}%)")
        print(f"  rank changes, cubic vs linear      : "
              f"{d['rank_change_cubic'][0]} of {d['rank_change_cubic'][1]} configurations")
        print(f"  {'algorithm':<12}{'campaign':>10}{'cubic':>10}")
        base = d["linear"]["meanrank"]
        for a in sorted(base, key=lambda x: base[x]):
            print(f"  {NAME.get(a,a):<12}{base[a]:10.2f}"
                  f"{d['cubic']['meanrank'][a]:10.2f}")
        for key in ("cubic",):
            st, p, L = d[key]["friedman"]
            print(f"  Friedman ({key}): chi2 = {st:.2f}, p = {p:.3e}, "
                  f"{L} configurations")
        print("  Wilcoxon, GNN-LX-SSA vs each competitor under the cubic ramp:")
        for a, v in sorted(d["cubic"]["wilcoxon"].items(),
                           key=lambda kv: kv[1]["p"]):
            print(f"    {NAME.get(a,a):<12} n={v['n']:3d}  R+={v['Rplus']:7.1f} "
                  f"R-={v['Rminus']:7.1f}  p={v['p']:.3e}  r={v['r']:+.3f}")


if __name__ == "__main__":
    main()
