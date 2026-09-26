"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory.  Every buffer crosses the C ABI as a
64-bit address, so addresses are declared ``c_int64``; ``c_int`` truncates
them and the shim segfaults on a buffer above 4 GiB.

Each declaration is checked for arity at import time: a short ``argtypes``
list is *silently* zero-filled by ctypes, so a missing 64-bit slot turns
into a truncated address rather than an error.
"""

import ctypes
import os
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-num2fraction.so"

_I = ctypes.c_int64
_D = ctypes.c_double

# symbol -> (restype, argtypes)
_SIGNATURES = {
    "n2f_convert_range": (None, [_I, _I, _I, _I, _I, _I, _I]),
    "n2f_convert_value": (ctypes.c_int32, [_D, _I, _I]),
    "n2f_precision": (_D, [_D]),
    "n2f_precision_array": (None, [_I, _I, _I]),
    "n2f_gcd": (_D, [_D, _D]),
    "n2f_gcd_array": (None, [_I, _I, _I, _I]),
    "n2f_js_remainder_array": (None, [_I, _I, _I, _I]),
    "n2f_js_round_array": (None, [_I, _I, _I]),
}

# Every symbol takes this many arguments according to src/kernels.mojo.  A
# mismatch between this table and the built library is a build bug, not
# something to paper over at call time.
_ARITY = {
    "n2f_convert_range": 7,
    "n2f_convert_value": 3,
    "n2f_precision": 1,
    "n2f_precision_array": 3,
    "n2f_gcd": 2,
    "n2f_gcd_array": 4,
    "n2f_js_remainder_array": 4,
    "n2f_js_round_array": 3,
}


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    for name, (restype, argtypes) in _SIGNATURES.items():
        try:
            fn = getattr(lib, name)
        except AttributeError as exc:  # pragma: no cover - build mismatch
            raise RuntimeError(f"{_LIB_PATH} is missing symbol {name!r}") from exc
        if len(argtypes) != _ARITY[name]:  # pragma: no cover - build mismatch
            raise RuntimeError(
                f"arity table for {name!r} says {_ARITY[name]}, "
                f"argtypes has {len(argtypes)}"
            )
        fn.restype = restype
        fn.argtypes = argtypes
    return lib


lib = _load()


def _addr(a) -> int:
    """Byte address of a C-contiguous buffer."""
    if not a.flags["C_CONTIGUOUS"]:
        raise ValueError("buffer must be C-contiguous")
    return a.ctypes.data


def as_f64(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)
