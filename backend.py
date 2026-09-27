"""
============================================================
BACKEND SELECTION  (GPU via CuPy, CPU via NumPy)
============================================================

Every other module imports `xp` from here and never imports
numpy/cupy directly. That single indirection is what makes the
same source run on a GPU node and on a laptop.

Usage
-----
    from backend import xp, asnumpy, get_rng, USING_GPU

Environment variables
---------------------
    WFLOP_BACKEND = gpu | cpu | auto      (default: auto)
    WFLOP_DTYPE   = float64 | float32     (default: float64)

Why float64 by default
----------------------
The penalty term (1 + 1e10*g)**2 reaches ~1e32 for a violation of
a few metres. float32 tops out at 3.4e38, so it does not overflow,
but it carries only ~7 significant digits: once a penalty is added
to a wake loss of order 1e3, the wake-loss information is lost
entirely. On an A100 the FP64 throughput is 9.7 TFLOP/s, so there
is no reason to trade accuracy here. float32 is offered for cards
with weak FP64 (consumer GPUs), where you should also lower the
penalty coefficient - see README.md.
============================================================
"""

import os

_MODE = os.environ.get("WFLOP_BACKEND", "auto").lower()
_DTYPE_NAME = os.environ.get("WFLOP_DTYPE", "float64").lower()

USING_GPU = False
xp = None

if _MODE in ("auto", "gpu"):
    try:
        import cupy as _cp
        # Touch the device: an installed CuPy with no visible GPU
        # raises here rather than at the first kernel launch.
        _cp.cuda.runtime.getDeviceCount()
        _cp.zeros(1)
        xp = _cp
        USING_GPU = True
    except Exception as exc:                      # noqa: BLE001
        if _MODE == "gpu":
            raise RuntimeError(
                "WFLOP_BACKEND=gpu was requested but CuPy/CUDA is unavailable: "
                f"{exc}"
            ) from exc
        xp = None

if xp is None:
    import numpy as _np
    xp = _np
    USING_GPU = False

DTYPE = xp.float32 if _DTYPE_NAME == "float32" else xp.float64


def asnumpy(a):
    """Bring an array back to host memory as a NumPy array."""
    if USING_GPU:
        import cupy as _cp
        return _cp.asnumpy(a)
    import numpy as _np
    return _np.asarray(a)


def get_rng(seed):
    """A generator object with the same API on both backends.

    NOTE ON REPRODUCIBILITY
    -----------------------
    Results are reproducible for a given (backend, seed), but a GPU
    run and a CPU run with the same seed do NOT produce identical
    numbers: the two libraries implement different bit generators,
    and the batched code draws random numbers in a different order
    from the original per-individual loops. Runs are statistically
    equivalent, not bit-identical. Any claim of equivalence should
    rest on the distribution over 30 seeds, not on one run.
    """
    if USING_GPU:
        import cupy as _cp
        return _cp.random.default_rng(seed)
    import numpy as _np
    return _np.random.default_rng(seed)


def device_info():
    if not USING_GPU:
        import numpy as _np
        return f"CPU / NumPy {_np.__version__} (dtype={xp.dtype(DTYPE).name})"
    import cupy as _cp
    props = _cp.cuda.runtime.getDeviceProperties(_cp.cuda.runtime.getDevice())
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    free_b, total_b = _cp.cuda.runtime.memGetInfo()
    return (f"GPU / CuPy {_cp.__version__} on {name} "
            f"({total_b / 2**30:.1f} GiB, {free_b / 2**30:.1f} GiB free, "
            f"dtype={_cp.dtype(DTYPE).name})")
