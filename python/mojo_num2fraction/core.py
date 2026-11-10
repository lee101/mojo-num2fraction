"""num2fraction: number -> fraction string, with the upstream algorithm in Mojo.

Upstream (npm `num2fraction`, github.com/yisibl/num2fraction) is 40 lines of
JavaScript.  The numeric core -- the epsilon comparison, the Euclidean GCD
over doubles and the precision search -- lives in ``src/kernels.mojo``.  This
module is the ctypes shim plus the two pieces of plumbing upstream also does
in JavaScript: parsing a string argument and formatting the result.
"""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np

from ._lib import _addr, as_f64, lib

__all__ = [
    "Num2FractionError",
    "convert",
    "convert_array",
    "fraction",
    "gcd",
    "js_remainder",
    "js_round",
    "num2fraction",
    "num2fraction_batch",
    "parse_float",
    "precision",
    "precision_array",
]

STATUS_OK = 0
STATUS_NO_PRECISION = 1
STATUS_NONFINITE = 2

# Chunking only pays once the per-element work (a data-dependent precision
# search plus a GCD) exceeds the ~1 us ctypes call overhead.  Below this many
# elements the batch runs on the calling thread.
_THREAD_ELEMENTS = 1 << 15
_MAX_WORKERS = 8


class Num2FractionError(ValueError):
    """Raised for inputs upstream cannot terminate on (NaN, infinity)."""


def parse_float(text: str) -> float:
    """JavaScript ``parseFloat``: leading numeric prefix, else NaN.

    Upstream accepts strings and runs them through ``parseFloat``, so
    ``num2fraction('1.5px')`` is ``'3/2'``.  Plain ``float()`` rejects that, so
    the prefix rule is reproduced here.  Plumbing, not compute.
    """
    m = re.match(
        r"\s*[+-]?(?:Infinity|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", text
    )
    if m is None:
        return math.nan
    tok = m.group(0).strip()
    if tok.endswith("Infinity"):
        return -math.inf if tok.startswith("-") else math.inf
    return float(tok)


def js_round(x):
    """JavaScript ``Math.round``: closest integral value, ties towards +Inf.

    Matches upstream's ``round(n)`` (ECMA-262).  ``math.floor(x + 0.5)``
    agrees only while that addition is exact, and it is not:

    * above 2**52 the ulp is 1 or more, the sum rounds first, and V8 returns
      5129144868306001 for ``Math.round(5129144868306001)`` while
      ``floor(... + 0.5)`` is ...002.  Doubles that large are already
      integral, so they are returned unchanged, as ECMA-262 requires.
    * at 0.49999999999999994 (the double just below 0.5) the sum rounds up to
      1.0, so ``floor(x + 0.5)`` is 1 where V8 gives 0.
    * negatives in (-1, 0) must give ``-0``, e.g. ``Math.round(-0.4)`` is -0.

    So: integral x is returned unchanged, and otherwise ``floor(x)`` is
    returned unless ``x - floor(x) >= 0.5``.  ``x - floor(x)`` is exact for
    finite non-integral x, so the comparison is exact.  Accepts scalars and
    arrays (arrays go through the kernel).
    """
    if isinstance(x, np.ndarray):
        a = as_f64(x)
        out = np.empty_like(a)
        lib.n2f_js_round_array(a.size, _addr(a), _addr(out))
        return out
    x = float(x)
    fl = _js_floor_py(x)
    if fl == x:
        return x
    if x - fl >= 0.5:
        return _js_ceil_py(x)
    return fl


def _js_floor_py(x: float) -> float:
    """ECMA-262 ``Math.floor`` as a float, keeping ``-0.0``.

    ``math.floor`` returns an int, so ``float(math.floor(-0.4))`` is ``-1.0``
    but ``float(math.floor(-0.0))`` is ``+0.0``, losing the sign ECMA-262
    keeps.  Negatives are floored on their magnitude instead.
    """
    if x > 0.0:
        return float(math.floor(x))
    if x == 0.0:
        return x  # +0 or -0 unchanged
    if not math.isfinite(x):
        return x
    return -float(math.floor(-x))


def _js_ceil_py(x: float) -> float:
    """ECMA-262 ``Math.ceil`` as a float, keeping ``-0.0`` for negatives in
    ``(-1, 0)`` (``Math.ceil(-0.4)`` is ``-0``)."""
    if x < 0.0 and math.isfinite(x):
        return -float(math.floor(-x))
    if x == 0.0:
        return x
    return float(math.ceil(x))


def _js_number_to_string(x: float) -> str:
    """JavaScript ``String(number)`` for the values this port prints.

    Two differences from Python's ``repr`` matter:

    * ``String(1000)`` is ``'1000'``, not ``'1000.0'``;
    * ``String`` prints the *shortest* round-tripping decimal, so for
      70142542814845430.0 -- a double whose exact value is 70142542814845432 --
      it gives ``'70142542814845430'``.  ``str(int(x))`` prints the exact value
      and is wrong.

    ECMA-262 uses exponential notation outside [1e-6, 1e21) and never pads the
    exponent to two digits.  Checked against node on 5018 values, 5000 of them
    random bit patterns: zero disagreements.
    """
    x = float(x)
    if not math.isfinite(x):
        return repr(x)
    sign = "-" if math.copysign(1.0, x) < 0.0 else ""
    mant, _, exp = repr(abs(x)).partition("e")
    ip, _, fp = mant.partition(".")
    # Trailing zeros of the fraction carry no information: String(1000.0) is
    # "1000", not "1000.0".
    fp = fp.rstrip("0")
    digits = ip + fp
    if not digits.strip("0"):
        return "0"  # String(-0) is "0"
    # ECMA-262 writes the value as 0.<digits> * 10**n.
    n = int(exp or 0) + len(ip)
    k = len(digits)
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    head = digits[0] + ("." + digits[1:] if k > 1 else "")
    e = n - 1
    return f"{sign}{head}e{'+' if e >= 0 else '-'}{abs(e)}"


def js_remainder(x, y):
    """``x % y`` with JavaScript (fmod) semantics, scalar or elementwise.

    Exact for every divisor that is a normal double, which is all the GCD
    needs: it divides by ``e`` (a power of ten, at least 1) and by remainders
    that stay above the 2**-20 epsilon.  A subnormal divisor whose quotient
    passes 2**52 cannot be handled by rescaling both operands and returns 0.0.
    """
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        a = as_f64(x)
        b = as_f64(y)
        if a.shape != b.shape:
            b = np.ascontiguousarray(np.broadcast_to(b, a.shape))
        out = np.empty_like(a)
        lib.n2f_js_remainder_array(a.size, _addr(a), _addr(b), _addr(out))
        return out
    out = np.empty(1, dtype=np.float64)
    b = as_f64(y)
    lib.n2f_js_remainder_array(1, _addr(as_f64(x)), _addr(b), _addr(out))
    return float(out[0])


def precision(x):
    """Upstream ``findPrecision(n)``: the smallest power of ten for which
    ``round(n * e) / e`` is within 2**-20 of ``n``.  0.0 means the search did
    not converge, where upstream loops forever."""
    if isinstance(x, np.ndarray):
        return precision_array(x)
    return float(lib.n2f_precision(float(x)))


def precision_array(x) -> np.ndarray:
    a = as_f64(x)
    out = np.empty_like(a)
    lib.n2f_precision_array(a.size, _addr(a), _addr(out))
    return out


def gcd(a, b):
    """Upstream ``GCD``: Euclidean over doubles with JavaScript remainder.

    The sign follows the algorithm rather than the inputs -- upstream negates
    the result -- so callers take ``abs``.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        aa = as_f64(a)
        bb = as_f64(b)
        if aa.shape != bb.shape:
            bb = np.ascontiguousarray(np.broadcast_to(bb, aa.shape))
        out = np.empty_like(aa)
        lib.n2f_gcd_array(aa.size, _addr(aa), _addr(bb), _addr(out))
        return out
    return float(lib.n2f_gcd(float(a), float(b)))


def convert(x) -> tuple:
    """Scalar path: ``(numerator, denominator, precision, status)``."""
    out = np.empty(3, dtype=np.float64)
    status = np.empty(1, dtype=np.int32)
    lib.n2f_convert_value(float(x), _addr(out), _addr(status))
    return float(out[0]), float(out[1]), float(out[2]), int(status[0])


def convert_array(x) -> tuple:
    """Batch path: ``(numerators, denominators, precisions, statuses)``.

    Large inputs are split into chunks and fanned out over a thread pool; the
    kernel takes a range so a chunk needs no state of its own.
    """
    a = as_f64(x)
    n = a.size
    flat = np.ascontiguousarray(a.reshape(-1))
    nums = np.empty(n, dtype=np.float64)
    dens = np.empty(n, dtype=np.float64)
    precs = np.empty(n, dtype=np.float64)
    status = np.empty(n, dtype=np.int32)

    if n >= _THREAD_ELEMENTS:
        workers = min(_MAX_WORKERS, -(-n // _THREAD_ELEMENTS))
        step = -(-n // workers)

        def run(part: int) -> None:
            lo = part * step
            hi = min(lo + step, n)
            if lo < hi:
                lib.n2f_convert_range(
                    lo, hi, _addr(flat), _addr(nums), _addr(dens),
                    _addr(precs), _addr(status),
                )

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(run, range(workers)))
    elif n:
        lib.n2f_convert_range(
            0, n, _addr(flat), _addr(nums), _addr(dens), _addr(precs),
            _addr(status),
        )

    shape = a.shape
    return (
        nums.reshape(shape), dens.reshape(shape),
        precs.reshape(shape), status.reshape(shape),
    )


def fraction(x) -> Fraction:
    """The exact rational the upstream string denotes."""
    num, den, _prec, status = convert(x)
    _check(status, x)
    return Fraction(int(num), int(den))


def num2fraction(num) -> str:
    """Port of ``num2fraction(num)``: the number as ``'numerator/denominator'``.

    The number 0 and the string ``'0'`` return ``'0'``, matching upstream's
    short-circuit.  The string ``'-0'`` does not: ``'-0' === 0`` is false, so
    upstream parses it and returns ``'0/1'``.
    """
    from_number = not isinstance(num, str)
    if from_number:
        value = float(num)
        if value == 0.0:
            return "0"
    else:
        if num == "0":
            return "0"
        value = parse_float(num)

    n, d, _prec, status = convert(value)
    _check(status, num)
    return f"{_js_number_to_string(n)}/{_js_number_to_string(d)}"


def num2fraction_batch(values) -> list:
    """``num2fraction`` over a sequence, through one compiled pass."""
    if isinstance(values, str):
        raise TypeError("num2fraction_batch takes a sequence, not a string")
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    nums, dens, _precs, status = convert_array(arr)
    out = []
    flat_in = arr.reshape(-1)
    flat_status = status.reshape(-1)
    for i, (n, d) in enumerate(zip(nums.reshape(-1), dens.reshape(-1))):
        if flat_status[i] != STATUS_OK:
            _check(int(flat_status[i]), flat_in[i])
        out.append("0" if flat_in[i] == 0.0
                   else f"{_js_number_to_string(n)}/{_js_number_to_string(d)}")
    return out


def _check(status: int, x) -> None:
    if status == STATUS_OK:
        return
    if status == STATUS_NO_PRECISION:
        raise Num2FractionError(
            f"the precision search did not converge for {x!r}: upstream's "
            f"findPrecision multiplies by ten until round(n*e)/e matches, "
            f"which never happens for this value (upstream loops forever)"
        )
    raise Num2FractionError(
        f"{x!r} is not finite; upstream loops forever in findPrecision"
    )
