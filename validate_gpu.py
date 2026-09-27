"""
============================================================
VALIDATION  -  run this FIRST on the GPU node
============================================================

    python validate_gpu.py

Checks, in order of importance:

 1. OBJECTIVE EQUIVALENCE, wind data sets I and II, against the
    original CPU implementation bundled beside this script
    (objective.py). Skipped with a loud message if it is missing -
    never silently "passed".

 2. SURROGATE EQUIVALENCE - the batched dense GNWM against the CPU
    sparse _GNWMSurrogate with identical weights (forward pass),
    plus a self-contained numerical gradient check of the manual
    backward pass.

 3. OPTIMIZER SANITY - all fourteen optimizers: shapes, monotone
    convergence curves, best score equal to the end of the curve,
    and for the GNN family that the reported best really is the
    exact objective of the reported layout.

 4. THROUGHPUT - objective evaluations per second across batch
    sizes, so you can see where the GPU saturates.
============================================================
"""

import sys
import time
import numpy as np

from backend import xp, asnumpy, device_info, USING_GPU
from objective_gpu import objective_batch, make_objective
from algorithms_gpu import build, ALGORITHMS
from surrogate_gpu import BatchedGNWM, build_graphs, SDTYPE
import gnn_algorithms_gpu  # noqa: F401  (registers the GNN optimizers)

# Every failed check is recorded here; the script exits non-zero if any
# failed, so `python validate_gpu.py || exit 1` really stops a job.
FAILURES = []


def verdict(ok, section):
    if not ok:
        FAILURES.append(section)
    return "PASS" if ok else "FAIL"


print("=" * 74)
print("WFLOP GPU VALIDATION")
print(device_info())
print("=" * 74)

# ------------------------------------------------------------------
print("\n[0] GPU readiness")
if not USING_GPU:
    print("    running on the CPU backend - this check is informational only.")
    print("    On the GPU node it reports the device, its FP64 capability and")
    print("    whether the campaign's peak tensors fit in memory.")
else:
    import cupy as _cp
    dev = _cp.cuda.Device()
    props = _cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props["name"]
    name = name.decode() if isinstance(name, bytes) else name
    cc = f"{props['major']}.{props['minor']}"
    free_b, total_b = _cp.cuda.runtime.memGetInfo()
    print(f"    device            : {name}  (compute capability {cc})")
    print(f"    memory            : {total_b/2**30:.1f} GiB total, "
          f"{free_b/2**30:.1f} GiB free")
    print(f"    CuPy / CUDA       : {_cp.__version__} / "
          f"{_cp.cuda.runtime.runtimeGetVersion()}")
    print(f"    objective dtype   : {xp.dtype(xp.zeros(1).dtype).name}"
          "   (float64 is intended: the 1e10 penalty needs the range)")

    # peak surrogate tensor, campaign settings
    R, POP, MODELS, F, N = 30, 30, 5, 64, 18
    item = 4 if str(SDTYPE).endswith("float32'>") else 8
    msg = lambda g, m: R * m * g * N * N * (2 * F + 5) * item / 2**30
    print(f"    peak GNN tensor   : {msg(2*POP, 1):.2f} GiB screening "
          f"(no ensemble), {msg(2*POP, MODELS):.2f} GiB with the 5-model "
          f"ensemble")
    print(f"    peak objective    : "
          f"{21 * R * POP * 24 * N * 8 / 2**30:.2f} GiB "
          f"(auto-chunked to WFLOP_MEM_BUDGET_MB)")
    head = msg(2 * POP, MODELS) * 3
    if head > total_b / 2**30 * 0.8:
        print("    -> WARNING: the UQ variants may not fit. Lower n_models, "
              "hidden, or NUM_RUNS.")
    else:
        print("    -> PASS (fits with headroom on this device)")

    a = xp.zeros((512, 512), dtype=xp.float64)
    b = (a + 1.0) @ (a + 2.0)
    _cp.cuda.runtime.deviceSynchronize()
    print(f"    fp64 kernel smoke : "
          f"{verdict(float(b[0, 0]) == 1024.0, '[0] fp64 kernel')}")

# ------------------------------------------------------------------
print("\n[1] Objective equivalence, wind data sets I and II")
try:
    import objective as ORIG
except Exception as exc:                                   # noqa: BLE001
    ORIG = None
    print(f"    SKIPPED - could not import the CPU objective.py ({exc})")

if ORIG is not None:
    rng = np.random.default_rng(7)
    worst = 0.0
    for ds in (1, 2):
        for n, radius in [(5, 500), (9, 500), (14, 750), (18, 1000)]:
            X = rng.uniform(-radius, radius, (8, 2 * n))
            if hasattr(ORIG, "scenario_objective"):
                ref = np.array([ORIG.scenario_objective(x, radius, dataset=ds)
                                for x in X])
            elif ds == 1:
                ref = np.array([ORIG.scenario1_objective(x, radius) for x in X])
            else:
                continue
            got = asnumpy(objective_batch(X, radius, dataset=ds))
            rel = np.abs(ref - got) / np.maximum(np.abs(ref), 1e-12)
            worst = max(worst, float(rel.max()))
        print(f"    data set {ds}: worst relative error so far {worst:.3e}")
    print(f"    -> {verdict(worst < 1e-10, '[1] objective equivalence')} "
          f"(differences are float summation order, not physics)")

# ------------------------------------------------------------------
print("\n[2] GNWM surrogate")
try:
    import algorithms as CPU_ALG
except Exception as exc:                                   # noqa: BLE001
    CPU_ALG = None
    print(f"    forward equivalence SKIPPED - no CPU algorithms.py ({exc})")

if CPU_ALG is not None and hasattr(CPU_ALG, "_GNWMSurrogate"):
    CPU_ALG.set_active_dataset(1)
    rng = np.random.default_rng(0)
    n, r, F, L = 7, 500.0, 16, 2
    pos = rng.uniform(-r * 0.7, r * 0.7, (n, 2))

    Xc, src, dst, Ec = CPU_ALG._GNWMSurrogate.build_graph(pos, r)
    thetas = xp.asarray(np.deg2rad(np.arange(0, 360, 15) + 7.5))
    omega = xp.asarray(CPU_ALG._OMEGA)
    Xg, Eg, mg = build_graphs(xp.asarray(pos)[None, None], r, omega, thetas,
                              float(CPU_ALG._ALPHA_CONE), float(CPU_ALG._AJ),
                              CPU_ALG._WTK, CPU_ALG._WTR, float(CPU_ALG._PSIBAR))

    cs = CPU_ALG._GNWMSurrogate(hidden=F, layers=L, seed=3)
    gs = BatchedGNWM(1, hidden=F, layers=L, seed=3)
    for k in gs.p:
        v = np.asarray(cs.p[k])
        gs.p[k] = xp.asarray(v.reshape((1,) + v.shape) if v.ndim > 1
                             else v.reshape(1, -1), dtype=SDTYPE)
    Pc, gc, _ = cs.forward(Xc, src, dst, Ec)
    Pg, gg, _ = gs.forward(Xg, Eg, mg)
    dp = abs(Pc - float(asnumpy(Pg)[0, 0])) / max(abs(Pc), 1e-9)
    dg = float(np.abs(asnumpy(gg)[0, 0] - gc).max())
    print(f"    edges: cpu {len(src)} / gpu {int(asnumpy(mg).sum())}")
    print(f"    power head relative diff {dp:.2e} | direction head max diff {dg:.2e}")
    print(f"    -> {verdict(dp < 1e-5 and dg < 1e-4, '[2] surrogate forward')} "
          f"(float32 surrogate precision)")

# gradient check (self-contained)
rng = np.random.default_rng(1)
n, r, F, L, G = 6, 500.0, 8, 2, 3
pos = rng.uniform(-r * 0.7, r * 0.7, (G, n, 2))
thetas = xp.asarray(np.deg2rad(np.arange(0, 360, 15) + 7.5))
import objective_gpu as OBJ
Xg, Eg, mg = build_graphs(xp.asarray(pos)[None], r, OBJ.OMEGA_1, thetas,
                          float(np.arctan(OBJ.K)),
                          float(1 - np.sqrt(1 - OBJ.CT)), OBJ.K, OBJ.R, 13.0)
gs = BatchedGNWM(1, hidden=F, layers=L, seed=5)
target = xp.asarray(rng.normal(5, 1, (1, G)))
P, gout, cache = gs.forward(Xg, Eg, mg, need_cache=True)
grads = gs.backward(cache, 2.0 * (P - target), None)


def _loss():
    p, _, _ = gs.forward(Xg, Eg, mg)
    return float(xp.sum((p - target) ** 2))


eps, worst_g = 1e-4 if SDTYPE == xp.float32 else 1e-6, 0.0
for name in ("Win", "W10", "U10", "V2", "bin", "d2"):
    idx = (0,) * gs.p[name].ndim
    a = float(grads[name][idx])
    old = float(gs.p[name][idx])
    gs.p[name][idx] = old + eps
    lp = _loss()
    gs.p[name][idx] = old - eps
    lm = _loss()
    gs.p[name][idx] = old
    num = (lp - lm) / (2 * eps)
    worst_g = max(worst_g, abs(a - num) / max(abs(num), 1e-6))
tol = 5e-2 if SDTYPE == xp.float32 else 1e-5
print(f"    manual backprop vs numerical gradient: worst relative {worst_g:.2e}")
print(f"    -> {verdict(worst_g < tol, '[2] surrogate gradient')}"
      + ("  (loose float32 tolerance: the finite difference itself is only"
         " good to ~1e-2 here. For a strict check re-run with"
         " WFLOP_SURROGATE_DTYPE=float64, where this lands near 1e-6.)"
         if SDTYPE == xp.float32 else ""))

# ------------------------------------------------------------------
print("\n[3] Optimizer sanity  (4 runs x 8 individuals x 5 iterations)")
radius, n = 500, 7
_f = make_objective(radius)
_rows = [0]


def f(X):
    """The objective, counting every layout actually computed."""
    _rows[0] += int(X.shape[0]) if X.ndim > 1 else 1
    return _f(X)


all_ok = True
for name in ALGORITHMS:
    kw = dict(n_pretrain=8, hidden=16, mp_layers=2, n_models=3) \
        if name.startswith("GNN") else {}
    alg = build(name, **kw)
    _rows[0] = 0
    bx, bf, cv, ne = alg.optimize(f, 2 * n, -radius, radius, 4, 8, 5, seed=1)
    bxn, bfn, cvn = asnumpy(bx), asnumpy(bf), asnumpy(cv)
    shapes = bxn.shape == (4, 2 * n) and cvn.shape == (4, 6)
    monotone = bool(np.all(np.diff(cvn, axis=1) <= 1e-9))
    matches = bool(np.allclose(cvn[:, -1], bfn, rtol=1e-9, atol=1e-6))
    exact = bool(np.allclose(asnumpy(objective_batch(bxn, radius)), bfn,
                             rtol=1e-9, atol=1e-6))
    # counters: ObjectiveCalls must equal the layouts really computed, and
    # Evaluations (what the optimizer used) can never exceed them
    ev = np.broadcast_to(np.atleast_1d(asnumpy(ne)), (4,)).astype(float)
    oc = np.broadcast_to(np.atleast_1d(asnumpy(
        getattr(alg, "n_objective_calls", ne))), (4,)).astype(float)
    counted = bool(oc.sum() == _rows[0] and np.all(ev <= oc))
    ok = shapes and monotone and matches and exact and counted
    all_ok &= ok
    print(f"    {name:12s} evals/run={int(np.mean(ev)):6d}  "
          f"shapes={'ok' if shapes else 'BAD'}  "
          f"monotone={monotone}  best==curve_end={matches}  "
          f"best==f(x)={exact}  counters={counted}  {'PASS' if ok else 'FAIL'}")
print(f"    -> {verdict(all_ok, '[3] optimizer sanity')}")

# ------------------------------------------------------------------
print("\n[3b] GNN family structure")
# GNN-QA-SSA and GNN-QA-SSA-UQ are derived code, not ports. The guarantee that
# makes them defensible is that they differ from the LX variants ONLY at the
# intended points - the candidate operator and the UQ flag - and share one
# implementation everywhere else. This check enforces that mechanically, so
# the four cannot drift apart in a later edit.
import inspect

import gnn_algorithms_gpu as G

_fam = {"GNNLXSSA": ("laplace", False), "GNNQASSA": ("quadratic", False),
        "GNNLXSSA_UQ": ("laplace", True), "GNNQASSA_UQ": ("quadratic", True)}
_base = G._GNNSalpBase
_struct_ok = True
for _n, (_op, _uq) in _fam.items():
    _c = G.ALGORITHMS[_n]
    shared = _c.optimize is _base.optimize
    own = {k for k, v in vars(_c).items()
           if not k.startswith("__") and k not in ("name", "operator", "uq")}
    ok = (shared and not own and _c.operator == _op and _c.uq is _uq
          and issubclass(_c, _base))
    print(f"    {_n:13} operator={_c.operator:9} uq={str(_c.uq):5} "
          f"shares base optimize={shared}  extra overrides={sorted(own) or 'none'}"
          f"  {'PASS' if ok else 'FAIL'}")
    _struct_ok &= ok
_lines = len(inspect.getsource(_base.optimize).splitlines())
print(f"    one shared implementation of {_lines} lines; the four subclasses "
      f"add nothing but two flags")
print(f"    -> {verdict(_struct_ok, '[3b] GNN structure')}")


# ------------------------------------------------------------------
print("\n[4] Objective edge cases")
# The equivalence check in [1] uses random layouts, which never land on the
# geometric boundaries where a wake model is most likely to be wrong. These
# cases put the objective exactly on them.
_R, _MIN = 38.5, 308.0
edge_fail = []


def _check(label, X, radius, expect):
    """expect: 'penalty', 'finite', or a reference value to match."""
    v = np.atleast_1d(asnumpy(objective_batch(xp.asarray(np.atleast_2d(X)),
                                              radius)))
    ok = np.all(np.isfinite(v))
    if expect == "penalty":
        ok = ok and np.all(v > 1e9)
    elif expect == "finite":
        ok = ok and np.all(v < 1e9)
    else:
        ok = ok and np.allclose(v, expect, rtol=1e-10, atol=1e-10)
    print(f"    {label:<46} {'PASS' if ok else 'FAIL'}   {v[0]:.6g}")
    if not ok:
        edge_fail.append(label)
    return v[0]


# coincident turbines: zero distance must not produce NaN or Inf
_check("two turbines at the same point -> penalty",
       np.array([0.0, 0.0, 0.0, 0.0]), 500, "penalty")

# exactly at the spacing constraint: feasible by >=, not by >
_check("spacing exactly 308 m (constraint boundary) -> feasible",
       np.array([-_MIN / 2, 0.0, _MIN / 2, 0.0]), 500, "finite")
_check("spacing 308 m minus 1 mm -> penalty",
       np.array([-_MIN / 2, 0.0, _MIN / 2 - 1e-3, 0.0]), 500, "penalty")

# exactly on the farm boundary
_check("turbine exactly on the boundary circle -> feasible",
       np.array([500.0, 0.0, 0.0, 400.0]), 500, "finite")
_check("turbine 1 mm outside the boundary -> penalty",
       np.array([500.0 + 1e-3, 0.0, 0.0, 400.0]), 500, "penalty")

# wake magnitude must fall as the pair separates. With 24 wind directions no
# pair is ever exactly wake-free, so the check is monotonicity, not zero.
far = _check("far-apart pair -> near-zero wake loss",
             np.array([0.0, -900.0, 0.0, 900.0]), 1000, "finite")
near = float(np.atleast_1d(asnumpy(objective_batch(
    xp.asarray(np.array([[0.0, -160.0, 0.0, 160.0]])), 1000)))[0])
ok = far < near and far < 1e-2
print(f"    {'wake loss grows as the pair closes':<46} "
      f"{'PASS' if ok else 'FAIL'}   {far:.3g} < {near:.3g}")
if not ok:
    edge_fail.append("wake monotonicity")

# no division by zero anywhere on a legal, widely-spread layout
with np.errstate(all="raise"):
    _check("wide-spread layout, no arithmetic warnings",
           np.array([-700.0, -700.0, 700.0, 700.0]), 1000, "finite")

# largest campaign case, batched, at full width
_big = np.random.default_rng(7).uniform(-1000, 1000, (900, 36))
_v = asnumpy(objective_batch(xp.asarray(_big), 1000))
print(f"    {'N=18, batch of 900, all finite':<46} "
      f"{'PASS' if np.all(np.isfinite(_v)) else 'FAIL'}")
if not np.all(np.isfinite(_v)):
    edge_fail.append("N=18 batch")

print(f"    -> {verdict(not edge_fail, '[4] objective edge cases')}"
      + (": " + ", ".join(edge_fail) if edge_fail else ""))


# ------------------------------------------------------------------
print("\n[5] Objective throughput (layouts evaluated per second)")
print(f"    {'N':>3} {'batch':>7} {'ms/call':>9} {'layouts/s':>12}")
BATCHES = (1, 30, 900, 4096, 16384, 65536) if USING_GPU else (1, 30, 900, 4096)
for n in (9, 18):
    for batch in BATCHES:
        X = np.random.default_rng(0).uniform(-500, 500, (batch, 2 * n))
        Xd = xp.asarray(X)
        objective_batch(Xd, 500)
        if USING_GPU:
            xp.cuda.runtime.deviceSynchronize()
        reps = 3 if batch > 4000 else 20
        t = time.perf_counter()
        for _ in range(reps):
            objective_batch(Xd, 500)
        if USING_GPU:
            xp.cuda.runtime.deviceSynchronize()
        dt = (time.perf_counter() - t) / reps
        print(f"    {n:3d} {batch:7d} {dt*1e3:9.2f} {batch/dt:12,.0f}")

print("\nThroughput should climb with batch size and then flatten. Run the")
print("campaign at or past that knee; raise it by putting more seeds in")
print("lockstep (NUM_RUNS in run_experiments_gpu.py).")
print("=" * 74)
if FAILURES:
    print(f"VALIDATION FAILED: {', '.join(FAILURES)}")
    print("=" * 74)
    sys.exit(1)
print("VALIDATION PASSED")
print("=" * 74)
