"""
============================================================
STRICT CuPy STAND-IN  (test utility, never imported at run time)
============================================================

Purpose: execute the package's GPU code path on a machine with no
GPU, so that everything except the CUDA kernels themselves is
actually run rather than merely reviewed.

This module masquerades as `cupy`. Put it on PYTHONPATH and the
backend selects the "GPU" branch:

    PYTHONPATH=tests/fake_cupy WFLOP_BACKEND=gpu python validate_gpu.py

It is deliberately STRICT: it exposes only names that real CuPy
provides, and every other attribute raises AttributeError. So if the
package ever reaches for something CuPy does not have - a NumPy-only
function, a NumPy scalar alias, a Generator method CuPy lacks - this
stub fails exactly where a real GPU would.

What it does NOT test: numerical behaviour of CUDA kernels, device
memory limits, and performance. Those still require real hardware.
============================================================
"""

import numpy as _np

__version__ = "13.0.0-stub"

# ---------------------------------------------------------------------------
# The array API surface real CuPy exposes and this package is allowed to use.
# Keep this list honest: adding a name here that CuPy lacks defeats the test.
# ---------------------------------------------------------------------------
_ALLOWED = [
    "abs", "all", "any", "arange", "arccos", "arctan", "argmin", "argmax",
    "argsort", "asarray", "ascontiguousarray", "broadcast_to", "clip",
    "concatenate", "cos", "cumsum", "diff", "dtype", "exp", "eye", "float32",
    "float64", "full", "isfinite", "isnan", "log", "matmul", "max", "maximum",
    "mean", "min", "minimum", "ndarray", "ones", "ones_like", "repeat",
    "reshape", "searchsorted", "sin", "sort", "sqrt", "square", "stack", "std",
    "sum", "take_along_axis", "transpose", "triu", "where", "zeros",
    "zeros_like", "allclose", "isclose", "int32", "int64", "bool_",
]

_self = __import__("sys").modules[__name__]
for _name in _ALLOWED:
    setattr(_self, _name, getattr(_np, _name))


def asnumpy(a):
    return _np.asarray(a)


def get_array_module(*args):            # real CuPy has this
    return _self


# ---------------------------------------------------------------------------
# cupy.random.Generator - only the methods real CuPy implements and the
# package is allowed to use.
# ---------------------------------------------------------------------------
class _Generator:
    _ALLOWED = ("random", "integers", "normal", "uniform", "standard_normal",
                "permutation", "shuffle")

    def __init__(self, seed=None):
        self._g = _np.random.default_rng(seed)

    def __getattr__(self, item):
        if item in self._ALLOWED:
            return getattr(self._g, item)
        raise AttributeError(
            f"cupy.random.Generator has no attribute {item!r} "
            "(this stub only exposes what real CuPy implements)")


class _RandomModule:
    Generator = _Generator

    @staticmethod
    def default_rng(seed=None):
        return _Generator(seed)


random = _RandomModule()


# ---------------------------------------------------------------------------
# cupy.cuda - device handling and the runtime calls the package makes
# ---------------------------------------------------------------------------
class _Runtime:
    _synced = 0
    _n_devices = int(__import__("os").environ.get("WFLOP_FAKE_GPUS", "1"))

    @classmethod
    def getDeviceCount(cls):
        return cls._n_devices

    @staticmethod
    def getDevice():
        return _Device._current

    @staticmethod
    def getDeviceProperties(dev):
        return {"name": b"NVIDIA A100-SXM4-40GB (stub)", "major": 8, "minor": 0}

    @staticmethod
    def memGetInfo():
        gib = 1024 ** 3
        return (34 * gib, 40 * gib)

    @classmethod
    def deviceSynchronize(cls):
        cls._synced += 1

    @staticmethod
    def runtimeGetVersion():
        return 12030


class _Device:
    _current = 0

    def __init__(self, dev_id=None):
        self.id = _Device._current if dev_id is None else int(dev_id)

    def use(self):
        _Device._current = self.id
        return self

    def __enter__(self):
        self.use()
        return self

    def __exit__(self, *a):
        return False


class _CudaModule:
    Device = _Device
    runtime = _Runtime


cuda = _CudaModule()


def __getattr__(name):
    raise AttributeError(
        f"cupy has no attribute {name!r} in this strict stub. Either real CuPy "
        "lacks it too - in which case the package must not use it - or add it "
        "to _ALLOWED after checking the CuPy API reference."
    )
