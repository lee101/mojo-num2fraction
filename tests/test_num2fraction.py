"""Parity tests for mojo-num2fraction.

Ground truth is the real npm ``num2fraction`` run under node; the fixture
``tests/upstream_vectors.json`` was produced by ``tests/upstream_vectors.js``
from the upstream ``index.js``.  The expected strings are therefore upstream
output, not this port's output.

A second, independent reference is implemented in pure Python below and
checked against the same fixture, so a shared misreading of the algorithm
would have to be made twice to go unnoticed.
"""

import math
import random
import struct

import numpy as np
import pytest

import mojo_num2fraction as m

# The upstream README's own table, plus the cases its changelog calls out.
# Verified against node running the real module.
README_CASES = [
    (0, "0"),
    (0.2, "1/5"),
    (1.1, "11/10"),
    (1.2, "6/5"),
    (1.3, "13/10"),
    (1.4, "7/5"),
    (1.5, "3/2"),
    (2, "2/1"),
    (2.1, "21/10"),
    (3, "3/1"),
    (2.555, "511/200"),
    (8.36, "209/25"),
    (7 / 9, "7/9"),
    (8 / 9, "8/9"),
    (512 / 999, "512/999"),
    ((2 * math.pi / 3) / math.pi, "2/3"),
    ((8 * 5) / (4 / 2), "20/1"),
]

EPSILON = 9.5367432e-7  # upstream's literal, exactly 2**-20


# ---------------------------------------------------------------- reference


def ref_js_round(x: float) -> float:
    if math.floor(x) == x:
        return x
    return float(math.floor(x + 0.5))


def ref_js_remainder(x: float, y: float) -> float:
    return math.fmod(x, y)


def ref_gcd(a: float, b: float) -> float:
    x, y = a, b
    guard = 0
    while abs(y) > EPSILON:
        if guard > 128:
            return 0.0
        x, y = y, ref_js_remainder(x, y)
        guard += 1
    return x


def ref_precision(n: float) -> float:
    e = 1.0
    for _ in range(400):
        if abs(ref_js_round(n * e) / e - n) <= EPSILON:
            return e
        e *= 10.0
        if not math.isfinite(e):
            break
    raise ValueError("no convergence")


def ref_num2fraction(x: float) -> str:
    if x == 0.0:
        return "0"
    e = ref_precision(x)
    number = x * e
    g = abs(ref_gcd(number, e))
    num = ref_js_round(number / g)
    den = ref_js_round(e / g)
    return f"{_ref_js_string(num)}/{_ref_js_string(den)}"


def _ref_js_string(x: float) -> str:
    if not math.isfinite(x):
        return repr(x)
    sign = "-" if math.copysign(1.0, x) < 0.0 else ""
    mant, _, exp = repr(abs(x)).partition("e")
    ip, _, fp = mant.partition(".")
    fp = fp.rstrip("0")
    digits = ip + fp
    if not digits.strip("0"):
        return "0"
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


# ------------------------------------------------------------------ tests


def test_readme_examples(upstream_vectors):
    for value, expected in README_CASES:
        assert m.num2fraction(value) == expected
        assert m.num2fraction(value) == ref_num2fraction(value)
    # the fixture agrees with the README table too, so it is the real thing
    table = {float(c["in"]): c["out"] for c in upstream_vectors["cases"]}
    for value, expected in README_CASES:
        assert table[value] == expected


def test_matches_every_upstream_vector(upstream_vectors):
    """1616 values produced by node running the real module."""
    bad = []
    for case in upstream_vectors["cases"]:
        got = m.num2fraction(float(case["in"]))
        if got != case["out"]:
            bad.append((case["in"], got, case["out"]))
    assert not bad, f"{len(bad)} mismatches, first: {bad[:5]}"


def test_python_reference_reproduces_the_fixture(upstream_vectors):
    """The independent reference must agree with upstream too, or it is no use
    as a second opinion on the kernel."""
    bad = [
        c for c in upstream_vectors["cases"]
        if ref_num2fraction(float(c["in"])) != c["out"]
    ]
    assert not bad, f"reference disagrees with upstream on {len(bad)}: {bad[:5]}"


def test_batch_api_matches_upstream_and_keeps_shape(upstream_vectors):
    values = np.array([float(c["in"]) for c in upstream_vectors["cases"]])
    expected = [c["out"] for c in upstream_vectors["cases"]]
    assert m.num2fraction_batch(values) == expected
    square = values[: 88 * 18].reshape(88, 18)
    assert m.num2fraction_batch(square) == expected[: 88 * 18]


def test_batch_threads_and_serial_agree():
    """The batch path chunks over a thread pool above a threshold; the chunking
    must not change a single answer."""
    rng = np.random.default_rng(20260926)
    values = rng.uniform(0.05, 9.95, size=40_000)
    assert values.size >= (1 << 15)  # the threaded path really was taken
    batch = m.num2fraction_batch(values)
    assert batch[:500] == [m.num2fraction(v) for v in values[:500]]


def test_random_differential_against_reference():
    rng = random.Random(4242)
    values = []
    for _ in range(3000):
        mag = rng.randint(-8, 12)
        digits = rng.randint(1, 17)
        v = float(f"{rng.uniform(0.1, 9.9):.{digits}f}") * (10.0 ** mag)
        values.append(-v if rng.random() < 0.5 else v)
    values += [2.0 ** k for k in range(-60, 60)]
    values += [10.0 ** k for k in range(-20, 21)]
    for v in values:
        got, want = m.num2fraction(v), ref_num2fraction(v)
        assert got == want, f"{v!r}: {got} != {want}"


def test_js_remainder_is_exact_fmod():
    """JavaScript's % is fmod.  Mojo's own % is a *floored* modulo and Python's %
    is floored too, so this is exactly where a plausible port goes wrong.

    Compared with rtol=0, atol=0: fmod is exactly specified by IEEE-754, so
    any difference is a bug rather than a rounding tolerance.

    It is exact for every finite pair, including subnormals and quotients past
    2**53; the slow path is exercised by the next test.
    """
    rng = random.Random(99)
    pairs = []
    while len(pairs) < 4000:
        a = math.ldexp(rng.uniform(-1, 1), rng.randint(-1080, 1020))
        b = math.ldexp(rng.uniform(-1, 1), rng.randint(-1080, 1020))
        if b == 0.0:
            continue
        pairs.append((a, b))  # both operands anywhere, subnormals included
    pairs += [
        (1e300, 1.0), (1.0, 1e300), (1e308, 7.0), (1e21, 7.0),
        (-5.5, 7.0), (5.5, -7.0), (-0.0, 7.0), (2.0 ** 53 + 2.0, 1024.0),
        (1e-300, 1e300), (5e-324, 1e-300), (1e-310, 3.0),
    ]
    a = np.array([p[0] for p in pairs], dtype=np.float64)
    b = np.array([p[1] for p in pairs], dtype=np.float64)
    got = m.js_remainder(a, b)
    want = np.array([math.fmod(x, y) for x, y in pairs], dtype=np.float64)
    np.testing.assert_array_equal(got, want)
    # the floored modulo really would have been wrong for many of them
    assert sum(p[0] % p[1] != math.fmod(p[0], p[1]) for p in pairs[:400]) > 50


def test_js_remainder_handles_extreme_ratios():
    """Quotients past 2**53 -- including ones that overflow to infinity -- take
    the binary-reduction path instead of the fma one.  This is where a
    "scale both operands down" scheme silently fails, because the quotient of
    two operands scaled by the same power of two does not change."""
    cases = [
        (1e300, 5e-324), (1e300, 1.0), (1.0, 1e-300), (5e-324, 1e-300),
        (1e308, 7.0), (2.0 ** 1000, 3.0), (1e-310, 1e300), (1e300, 1e-310),
        (-1e300, 3.0), (1e300, -3.0), (2.0 ** 60, 2.0 ** -60),
    ]
    a = np.array([c[0] for c in cases], dtype=np.float64)
    b = np.array([c[1] for c in cases], dtype=np.float64)
    np.testing.assert_array_equal(
        m.js_remainder(a, b),
        np.array([math.fmod(x, y) for x, y in cases], dtype=np.float64),
    )
    # num2fraction reaches huge quotients through the GCD's first division
    assert m.num2fraction(1e300) == "1e+300/1"
    assert m.num2fraction(1e-310) == "0/1"


def test_js_round_ties_and_large_values():
    """Math.round is floor(x + 0.5) only while that addition is exact; the
    expectations below are node's."""
    values = [
        0.5, 1.5, 2.5, -0.5, -1.5, -2.5, -0.75, 0.75,
        4503599627370495.5, -4503599627370495.5,      # 2**52 boundary
        5129144868306001.0, -5129144868306001.0,      # node: unchanged
        1e21, -1e21, 1e22,
    ]
    want = [
        1.0, 2.0, 3.0, 0.0, -1.0, -2.0, -1.0, 1.0,
        4503599627370496.0, -4503599627370495.0,
        5129144868306001.0, -5129144868306001.0,
        1e21, -1e21, 1e22,
    ]
    got = m.js_round(np.array(values, dtype=np.float64))
    np.testing.assert_array_equal(got, np.array(want, dtype=np.float64))
    # floor(x + 0.5) -- the one-liner this replaces -- is wrong on the big ones
    assert math.floor(5129144868306001.0 + 0.5) != 5129144868306001.0


def test_epsilon_decides_the_precision_search():
    """The 2**-20 epsilon collapses 1e-7 to 0/1 and stretches 1e-6's search
    out to e = 1e6.  Every expectation here comes from node; change the
    epsilon constant and all of them move."""
    assert m.precision(1e-7) == 1.0
    assert m.num2fraction(1e-7) == "0/1"
    assert m.precision(5e-7) == 1.0
    assert m.precision(1e-6) == 1e6
    assert m.num2fraction(1e-6) == "1/1000000"
    assert m.precision(2e-6) == 1e6
    assert m.num2fraction(2e-6) == "1/500000"
    assert m.precision(1e-5) == 1e5
    # Math.round(0.5) is 1, so even one half needs a scale factor of ten
    assert m.precision(0.5) == 10.0
    assert m.precision(0.3333333333333333) == 1e6
    assert m.num2fraction(0.3333333333333333) == "1/3"


def test_gcd_matches_reference_including_its_sign():
    rng = random.Random(7)
    a = np.array([rng.uniform(-100, 100) for _ in range(2000)], dtype=np.float64)
    b = np.array([rng.uniform(0.5, 100) for _ in range(2000)], dtype=np.float64)
    np.testing.assert_array_equal(
        m.gcd(a, b),
        np.array([ref_gcd(x, y) for x, y in zip(a, b)], dtype=np.float64),
    )
    # upstream returns the running value, whose sign follows the path; the
    # caller takes abs()
    assert m.gcd(-5.0, 10.0) == -5.0
    assert abs(m.gcd(-5.0, 10.0)) == 5.0
    assert m.gcd(10.0, 4.0) == 2.0


def test_non_finite_inputs_raise_where_upstream_hangs(upstream_vectors):
    """Verified with `timeout 3 node`: Infinity, NaN and -Infinity each exit
    124, because Math.round(inf * e) / e is NaN forever."""
    assert set(upstream_vectors["nonterminating"]) == {
        "Infinity", "NaN", "-Infinity"
    }
    for bad_value in (math.inf, -math.inf, math.nan):
        with pytest.raises(m.Num2FractionError):
            m.num2fraction(bad_value)
        with pytest.raises(m.Num2FractionError):
            m.num2fraction(str(bad_value))


def test_number_and_string_entry_points_differ_for_minus_zero():
    """Upstream short-circuits on `num === 0`, which is true for the number
    -0 but not for the string '-0'.  Both answers come from node."""
    assert m.num2fraction(-0.0) == "0"
    assert m.num2fraction("-0") == "0/1"
    assert m.num2fraction(0.0) == "0"
    assert m.num2fraction("0") == "0"


def test_parse_float_matches_javascript_for_plumbing():
    # upstream's README: n2f('3em') === '3/1', n2f('1.5px') === '3/2'
    assert m.num2fraction("3em") == "3/1"
    assert m.num2fraction("1.5px") == "3/2"
    assert m.parse_float("  -12.5e2kg") == -1250.0
    assert m.parse_float("Infinity") == math.inf
    assert math.isnan(m.parse_float("abc"))
    with pytest.raises(m.Num2FractionError):
        m.num2fraction("abc")


def test_fraction_helper_returns_the_exact_rational():
    from fractions import Fraction
    assert m.fraction(0.5) == Fraction(1, 2)
    assert m.fraction(2.555) == Fraction(511, 200)
    assert m.fraction(0.1) == Fraction(1, 10)
    assert m.fraction(0) == Fraction(0, 1)


def test_convert_array_reports_status_and_shape():
    nums, dens, precs, status = m.convert_array(
        np.array([[0.5, 1.5], [-0.25, 2.0]], dtype=np.float64)
    )
    assert nums.shape == (2, 2)
    assert status.shape == (2, 2)
    assert status.max() == m.core.STATUS_OK
    np.testing.assert_array_equal(nums, np.array([[1.0, 3.0], [-1.0, 2.0]]))
    np.testing.assert_array_equal(dens, np.array([[2.0, 2.0], [4.0, 1.0]]))
    # node: precision(0.5) == precision(1.5) == 10, precision(-0.25) == 100
    # (Math.round(-0.25) is 0, whose distance from -0.25 is far past the
    # epsilon), and precision(2) == 1.
    np.testing.assert_array_equal(precs, np.array([[10.0, 10.0], [100.0, 1.0]]))


def test_buffers_above_four_gib(mmap_high):
    """A ctypes argtypes list one entry short is silent: the 64-bit address
    lands in a 32-bit slot and is truncated.  Small arrays pass; anything above
    4 GiB dies.  This buffer is mapped at a fixed high address."""
    a = mmap_high
    assert a.ctypes.data >= (1 << 32)
    a[0] = 0.5
    a[1] = 1.5
    a[a.size - 1] = 2.555
    nums, dens, precs, status = m.convert_array(a)
    assert status[a.size - 1] == 0
    assert (nums[0], dens[0]) == (1.0, 2.0)
    assert (nums[1], dens[1]) == (3.0, 2.0)
    assert (nums[a.size - 1], dens[a.size - 1]) == (511.0, 200.0)
    del nums, dens, precs, status


def test_random_doubles_from_raw_bit_patterns():
    """Bit-level coverage: nothing decimal about the inputs, so a kernel that
    quietly assumed a 'reasonable' magnitude would show up here."""
    rng = random.Random(2024)
    values = []
    for _ in range(2000):
        v = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
        if not math.isfinite(v) or v == 0.0:
            continue
        values.append(v)
    assert values
    for v in values:
        got, want = m.num2fraction(v), ref_num2fraction(v)
        assert got == want, f"{v!r}: {got} != {want}"
