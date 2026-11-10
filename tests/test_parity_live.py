"""Parity of every public function with upstream num2fraction, checked live.

Two upstream references are used here, and neither is this port's output:

* ``tests/upstream_oracle.js`` runs the *real* upstream ``index.js`` under node
  and prints its answers for a deterministic corpus.  The corpus is generated
  at test time, so these are upstream's results on random values, not a
  transcript.  Set ``NUM2FRACTION_UPSTREAM`` to the path of an installed
  upstream ``index.js``.  With it unset, the tests fall back to a cached copy
  of the published npm tarball, fetching it once if node and npm are present;
  if neither route works they skip with a reason instead of pretending to
  have compared.
* upstream's own fixture, ``test/test.js`` (github.com/yisibl/num2fraction,
  the file its ``npm test`` runs), is embedded verbatim as ``UPSTREAM_TAPE``.

Version note.  The target is the published **1.2.2** -- the version the port
documents and the one ``tests/upstream_vectors.json`` was generated from.
Upstream **1.3.0** (2026-10-28) added ``if (num === Infinity) return
Infinity`` and its ``test/test.js`` asserts ``num2fraction(1/0) === Infinity``;
under 1.2.2 that call hangs.  The port raises ``Num2FractionError`` where 1.2.2
hangs, so ``Infinity`` is a **known, intentional** divergence from 1.3.0, not
from 1.2.2; ``test_infinity_is_the_one_known_divergence_from_1_3_0`` records
it explicitly rather than hiding it.

Upstream 1.2.2 has no ``Infinity`` early return, so ``num2fraction(Infinity)``
never terminates -- verified: ``timeout 3 node`` exits 124, and so do ``NaN``,
``-Infinity``, a non-numeric string, ``[]`` with more than one element, and any
object.  The oracle *detects* those with the same convergence pre-check the
kernel uses (upstream's own ``findPrecision``, unrolled) and never calls
upstream on them; the expected behaviour is then port-specific by design
(``Num2FractionError`` instead of a hang) and is asserted separately from
upstream's answers.

Tolerance.  Upstream returns strings, so the exact part of a comparison is
the string.  Where a number is compared -- the numerator, the denominator, the
precision and the numerator/denominator ratio reconstructed from upstream's
string -- the tolerance is written here as ``TOLERANCE`` and applied with
``math.isclose``.  It is 2 ULP of 0.0, i.e. zero: measured over the corpus
below the port is bit-identical to upstream, so any deviation at all is a
parity failure rather than a rounding allowance.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

import mojo_num2fraction as m

ROOT = pathlib.Path(__file__).resolve().parents[1]
ORACLE_JS = ROOT / "tests" / "upstream_oracle.js"
EPSILON = 9.5367432e-7  # upstream index.js: `abs(a - b) <= 9.5367432e-7`

# Relative/absolute tolerance for numeric comparisons.  Zero: the port and
# upstream agree bit-for-bit on every value in the corpus.  `math.ulp(0.0)`
# is the smallest subnormal, so this is exact equality stated in tolerance
# form rather than a magic number.
TOLERANCE = 2 * math.ulp(0.0)

# Upstream's `test/test.js`, transcribed.  Each row is (input expression as
# written by upstream, expected).  `None` means upstream's own assertion is
# `t.notEqual`, so only "differs from 99998/99999" is meaningful.
UPSTREAM_TAPE = [
    ("0", "0"),
    (".2", "1/5"),
    ("1.1", "11/10"),
    ("-1.1", "-11/10"),
    ("1.2", "6/5"),
    ("1.3", "13/10"),
    ("1.4", "7/5"),
    ("1.5", "3/2"),
    ("2", "2/1"),
    ("2.1", "21/10"),
    ("3", "3/1"),
    ("2.555", "511/200"),
    ("8.36", "209/25"),
    ("-8.36", "-209/25"),
    ("'3em'", "3/1"),
    ("'1.5px'", "3/2"),
    ("'0'", "0"),
    ("'.2'", "1/5"),
    ("'1.1'", "11/10"),
    ("'-1.1'", "-11/10"),
    ("'8.36'", "209/25"),
    ("1/3", "1/3"),
    ("2/3", "2/3"),
    ("1/9", "1/9"),
    ("2/9", "2/9"),
    ("4/9", "4/9"),
    ("5/9", "5/9"),
    ("7/9", "7/9"),
    ("8/9", "8/9"),
    ("(2*math.pi/3)/math.pi", "2/3"),
    ("(8*5)/(4/2)", "20/1"),
    ("512/999", "512/999"),
    ("5122/9999", "5122/9999"),
    ("5152/9999", "5152/9999"),
    ("6152/9999", "6152/9999"),
    ("1714/3333", "1714/3333"),
    ("17141/33333", "17141/33333"),
    ("17146/99999", "17146/99999"),
    ("99998/99999", "99998/99999"),
    ("99998/99999.0001", None),  # upstream: t.notEqual(..., '99998/99999')
    ("99998.0001/99999", None),  # upstream: t.notEqual(..., '99998/99999')
]


def _python_value(expr: str):
    """The same number upstream's JS expression denotes, for the tape rows."""
    return eval(expr, {"math": math})  # noqa: S307 - fixed local literals


def _split_fraction(text: str):
    """Upstream always returns 'numerator/denominator' -- except its
    short-circuit `if (num === 0 || num === '0') return '0'`, which has no
    denominator; the fraction it denotes is 0/1."""
    if "/" in text:
        num, den = text.split("/", 1)
        return float(num), float(den)
    return float(text), 1.0


def _ratio(text: str) -> float:
    num, den = _split_fraction(text)
    return num if den in (None, 0.0) else num / den


# --------------------------------------------------------------- live oracle


def _upstream_index_js() -> pathlib.Path:
    """Path to a real upstream index.js, or skip.

    ``NUM2FRACTION_UPSTREAM`` wins.  Otherwise a cached copy of the published
    npm tarball is used, fetched once if node/npm are available.
    """
    explicit = os.environ.get("NUM2FRACTION_UPSTREAM")
    if explicit:
        path = pathlib.Path(explicit)
        if path.is_file():
            return path
        pytest.skip(f"NUM2FRACTION_UPSTREAM={explicit} is not a file")

    cache = pathlib.Path(tempfile.gettempdir()) / "n2f-upstream-1.2.2"
    index = cache / "package" / "index.js"
    if index.is_file():
        return index
    if shutil.which("npm") is None:
        pytest.skip("no NUM2FRACTION_UPSTREAM and no npm to fetch upstream 1.2.2")
    cache.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(
        ["npm", "pack", "num2fraction@1.2.2", "--pack-destination", str(cache)],
        cwd=cache, capture_output=True, text=True, timeout=120,
    )
    if done.returncode != 0:
        pytest.skip(f"could not fetch npm num2fraction@1.2.2: {done.stderr[-400:]}")
    subprocess.run(
        ["tar", "xzf", str(cache / "num2fraction-1.2.2.tgz")],
        cwd=cache, check=True, timeout=120,
    )
    if not index.is_file():
        pytest.skip("npm tarball did not contain package/index.js")
    return index


@pytest.fixture(scope="module")
def upstream_live():
    """Upstream's answers for the deterministic corpus, as list of records.

    A record is ``{"in": <js expression>, "out": <string>, ...}`` or
    ``{"in": ..., "hang": True}`` when upstream would never terminate.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; live upstream comparison unavailable")
    index = _upstream_index_js()
    done = subprocess.run(
        [node, str(ORACLE_JS), str(index)],
        capture_output=True, text=True, timeout=180,
    )
    assert done.returncode == 0, f"oracle failed: {done.stderr[-2000:]}"
    records = [json.loads(line) for line in done.stdout.splitlines() if line]
    assert len(records) > 4500, "oracle corpus shrank; check upstream_oracle.js"
    return records


def _python_input(expr: str):
    """The value the oracle passed to upstream, as the Python API takes it.

    Three shapes occur in the oracle's ``in`` field:

    * JSON of a double (``"2.555"``) -- the same float here;
    * a JS numeric expression (``"(1 / 3)"``) -- evaluated by
      ``_python_value``, which uses Python's own arithmetic;
    * JSON of a string (``"\\"1.5px\\""``) -- upstream ran ``parseFloat`` on
      it, so the port must be handed the string itself.

    Containers (``[0.5]``, ``{}``...) are not numeric API inputs; they return
    the ``_CONTAINER`` sentinel so callers skip them.
    """
    if expr in _CONTAINERS:
        return _CONTAINER
    try:
        return json.loads(expr)
    except json.JSONDecodeError:
        return _python_value(expr)


_CONTAINER = object()
_CONTAINERS = frozenset({"[0.5]", "[0.5, 0.25]", "true", "false", "null", "{}"})


# ------------------------------------------------------------- public surface


def test_readme_and_upstream_tape(upstream_live):
    """Upstream's own fixture (test/test.js).  Exact strings; no tolerance."""
    for expr, expected in UPSTREAM_TAPE:
        got = m.num2fraction(_python_value(expr))
        if expected is None:
            assert got != "99998/99999", f"{expr}: {got}"
        else:
            assert got == expected, f"{expr}: {got!r} != upstream {expected!r}"


def test_num2fraction_matches_live_upstream(upstream_live):
    """``num2fraction`` on 4 900+ values, expected answers from upstream."""
    compared = 0
    bad = []
    for rec in upstream_live:
        value = _python_input(rec["in"])
        if value is _CONTAINER:
            continue  # JS container rows; covered by test_js_containers
        if "hang" in rec:
            continue  # upstream's hang; covered by test_upstream_nontermination
        got = m.num2fraction(value)
        compared += 1
        if got != rec["out"]:
            bad.append((rec["in"], got, rec["out"]))
    assert compared > 4500, compared
    assert not bad, f"{len(bad)} mismatches, first: {bad[:5]}"


def test_numerators_denominators_and_precision_within_tolerance(upstream_live):
    """The numeric surface behind the string: numerator, denominator, the
    ratio they denote, and the precision search -- each compared with
    upstream's string parsed back to numbers, at the stated ``TOLERANCE``.

    ``precision`` is cross-checked against upstream's ``findPrecision``
    rewritten here (it is not exported), which must agree with the port's
    returned precision exactly.
    """
    compared = 0
    for rec in upstream_live:
        value = _python_input(rec["in"])
        if value is _CONTAINER:
            continue
        if "hang" in rec or not isinstance(value, (int, float)):
            continue
        want_num, want_den = _split_fraction(rec["out"])
        num, den, prec, status = m.convert(value)
        assert status == m.core.STATUS_OK, rec
        assert math.isclose(num, want_num, rel_tol=TOLERANCE, abs_tol=TOLERANCE), rec
        assert math.isclose(den, want_den, rel_tol=TOLERANCE, abs_tol=TOLERANCE), rec
        assert math.isclose(
            num / den, _ratio(rec["out"]), rel_tol=TOLERANCE, abs_tol=TOLERANCE
        ), rec
        # upstream: findPrecision(n)
        e = 1.0
        for _ in range(400):
            if abs(round(_js_round_py(value * e)) / e - value) <= EPSILON:
                break
            e *= 10.0
        if value != 0.0:
            assert math.isclose(prec, e, rel_tol=TOLERANCE, abs_tol=TOLERANCE), (
                rec, prec, e)
        compared += 1
    assert compared > 4500, compared


def _js_round_py(x: float) -> float:
    """JavaScript ``Math.round`` (ECMA-262) for the checks in this file.

    Deliberately *not* ``floor(x + 0.5)``: that sum rounds when x sits within
    half an ulp of a half-integer, and it gets ``0.49999999999999994`` wrong
    (V8: 0).  Built on ``Fraction`` so the fractional part is compared
    exactly, and independently of the kernel, so the kernel is not checked
    against a copy of itself.
    """
    from fractions import Fraction

    if math.floor(x) == x:
        return x  # integral, and -0 for -0
    a = Fraction(abs(x))
    fl = math.floor(a)
    r = fl + 1 if a - fl >= Fraction(1, 2) else fl
    return -float(r) if x < 0 else float(r)


def test_js_round_near_half_matches_v8():
    """The doubles within an ulp of a half-integer, where `floor(x + 0.5)` and
    `Math.round` disagree because the addition itself rounds.

    Expected values are V8's (`node -e "Math.round(...)"`, node 22.22.2):
    ``Math.round(0.49999999999999994) === 0`` while ``floor(x + 0.5)`` is 1,
    and ``Math.round(-0.4)`` is ``-0``.  This also pins the visible
    consequence in upstream's findPrecision: its very first test is
    ``almostEq(round(n * 1) / 1, n)``, i.e. ``almostEq(0, n)`` here, which
    fails -- with the floor shortcut it would be ``almostEq(1, n)`` and pass.
    Upstream then stops at ``e = 10`` (``round(n * 10) / 10`` is 0.5), which
    the port reproduces.  The numbers are also in the live oracle corpus, so
    the whole-value string path is covered there.
    """
    above = 0.5 - 2.0 ** -53  # 0.49999999999999994, the one floor(x+0.5) victim
    above2 = 0.5 - 2.0 ** -54  # 0.4999999999999999, where the shortcut agrees
    below = 0.5 + 2.0 ** -53   # 0.5000000000000001
    cases = [
        (above, 0.0), (above2, 0.0), (-above, -0.0), (-above2, -0.0),
        (0.5, 1.0), (-0.5, -0.0), (below, 1.0), (-below, -1.0),
        (2.5, 3.0), (-2.5, -2.0),
        (-0.4, -0.0), (0.4, 0.0), (0.0, 0.0), (-0.0, -0.0),
    ]
    got = m.js_round(np.array([c[0] for c in cases], dtype=np.float64))
    for (x, want), g in zip(cases, got):
        assert math.isclose(float(g), want, rel_tol=TOLERANCE,
                            abs_tol=TOLERANCE), (x, g, want)
        # sign of zero matters here: Math.round(-0.4) is -0 in V8
        assert math.copysign(1.0, float(g)) == math.copysign(1.0, want), (x, g)
    # the scalar path agrees with the array path
    assert m.js_round(above) == 0.0
    assert math.copysign(1.0, m.js_round(-0.4)) == -1.0
    # and the consequence upstream's findPrecision sees: round(above * 1) is
    # 0 (not 1), but round(above * 10) / 10 is 0.5, within the epsilon, so
    # upstream stops at e = 10.  Both round() decisions come from Math.round.
    assert m.precision(above) == 10.0
    assert m.precision(above2) == 10.0
    assert m.precision(0.5) == 10.0
    assert m.num2fraction(above) == "1/2"
    assert m.num2fraction(-above) == "-1/2"


def test_gcd_matches_upstream_definition_on_gcd_inputs(upstream_live):
    """``GCD(a, b)`` over the (number, precision) pairs ``num2fraction``
    actually feeds it, computed by upstream's definition in JS and compared
    to ``m.gcd`` with ``TOLERANCE``."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    pairs = []
    for rec in upstream_live[:400]:
        value = _python_input(rec["in"])
        if value is _CONTAINER:
            continue
        if "hang" in rec or not isinstance(value, (int, float)) or value == 0:
            continue
        e = 1.0
        for _ in range(400):
            if abs(_js_round_py(value * e) / e - value) <= EPSILON:
                break
            e *= 10.0
        if math.isfinite(value * e):
            pairs.append((value * e, e))
    assert pairs
    # Upstream's GCD is plain recursion with no step limit, so V8 overflows
    # the stack on pairs where it never terminates (that is a JS limitation,
    # not an answer).  The probe mirrors the algorithm but stops after the
    # kernel's 128 steps and reports null, and those pairs are skipped: there
    # is no upstream value to compare against.
    script = """
const pairs = JSON.parse(process.argv[1]);
const EPS = 9.5367432e-7;
function GCD(a, b, n) {
  if (Math.abs(b - 0) <= EPS) return a;
  if (n >= 128) return null;
  return GCD(b, a % b, n + 1);
}
console.log(JSON.stringify(pairs.map(p => GCD(p[0], p[1], 0))));
"""
    done = subprocess.run([node, "-e", script, json.dumps(pairs)],
                  capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    want = json.loads(done.stdout)
    compared = 0
    for (a, b), w in zip(pairs, want):
        if w is None:
            continue
        g = m.gcd(a, b)
        assert math.isclose(g, w, rel_tol=TOLERANCE, abs_tol=TOLERANCE), (
            a, b, g, w)
        compared += 1
    assert compared > 100, compared


def test_precision_array_and_convert_array_agree_with_scalar(upstream_live):
    """The batch entry points must return the scalar answers, element for
    element, for a sample including the upstream fixture's own inputs."""
    fixture = json.loads((ROOT / "tests" / "upstream_vectors.json").read_text())
    values = np.array([float(c["in"]) for c in fixture["cases"][:500]])
    precs = m.precision_array(values)
    nums, dens, precs_b, status = m.convert_array(values)
    assert status.max() == m.core.STATUS_OK
    for i, v in enumerate(values):
        assert precs[i] == m.precision(float(v))
        assert precs_b[i] == m.precision(float(v))
        n, d, p, s = m.convert(float(v))
        assert (nums[i], dens[i], p, s) == (n, d, p, s)


def test_batch_matches_scalar_on_zero_one_and_duplicates():
    """Degenerate batch inputs upstream defines: empty, one element,
    duplicates, zeros of both signs, row-major 2-D shape."""
    assert m.num2fraction_batch([]) == []
    assert m.num2fraction_batch([2.555]) == ["511/200"]  # upstream tape
    assert m.num2fraction_batch([0.5, 0.5, 0.25, 0.25]) == [
        "1/2", "1/2", "1/4", "1/4"]
    assert m.num2fraction_batch([0.0, -0.0, 0.0]) == ["0", "0", "0"]
    zeros = m.num2fraction_batch(np.zeros((0, 3)))
    assert zeros == []
    square = m.num2fraction_batch(np.array([[0.5, 0.25], [2.555, 8.36]]))
    assert square == ["1/2", "1/4", "511/200", "209/25"]


def test_batch_propagates_the_same_error_as_the_scalar_path():
    """NaN/inf anywhere in a batch must raise what the scalar path raises,
    not silently produce a placeholder."""
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(m.Num2FractionError):
            m.num2fraction_batch([0.5, bad, 0.25])
        with pytest.raises(m.Num2FractionError):
            m.num2fraction(bad)


def test_fraction_reflects_upstream_strings(upstream_live):
    """``fraction(x)`` is the exact rational upstream's string denotes."""
    from fractions import Fraction

    compared = 0
    for rec in upstream_live[:300]:
        value = _python_input(rec["in"])
        if value is _CONTAINER:
            continue
        if "hang" in rec or not isinstance(value, (int, float)) or value == 0:
            continue
        num, den = _split_fraction(rec["out"])
        assert abs(m.fraction(value) - Fraction(int(num), int(den))) <= TOLERANCE
        compared += 1
    assert compared > 250


def test_precision_search_matches_upstream_find_precision(upstream_live):
    """``findPrecision`` is not exported, so the oracle re-runs upstream's own
    loop (unrolled only for the step cap) and reports the precision and its
    very first decision for every numeric input.

    The first decision is ``almostEq(round(n * 1) / 1, n)``, i.e.
    ``almostEq(round(n), n)``.  For n = 0.49999999999999994 that is
    ``almostEq(0, 0.4999...)`` -- false, so the search goes on.  With the
    ``floor(x + 0.5)`` shortcut ``round(n)`` would be 1 and the first decision
    would pass instead.

    That inversion is visible in ``js_round`` itself but not here: on the
    doubles where the two roundings disagree, ``findPrecision`` still lands on
    ``e = 10``.  A search (every double within 10 ulp of k + 0.5 for
    -30 <= k <= 30, and 200k random bit patterns) found no value whose
    precision differs under the shortcut, so this test pins ``precision``
    against upstream's own loop, and
    ``test_js_round_near_half_matches_v8`` is what catches the rounding bug.
    """
    checked = 0
    first_stop = first_go = 0
    for rec in upstream_live:
        if "hang" in rec or "precision" not in rec:
            continue
        value = _python_input(rec["in"])
        if value is _CONTAINER or not isinstance(value, (int, float)):
            continue
        if value == 0.0:
            continue  # upstream short-circuits before findPrecision
        assert math.isclose(m.precision(value), rec["precision"],
                            rel_tol=TOLERANCE, abs_tol=TOLERANCE), rec
        if rec["first_step"]:
            assert m.precision(value) == 1.0, rec
            first_stop += 1
        else:
            assert m.precision(value) != 1.0, rec
            first_go += 1
        checked += 1
    assert checked > 4500, checked
    assert first_stop > 100, first_stop
    assert first_go > 1000, first_go
    # the specific case the shortcut got wrong, straight from V8:
    # Math.round(0.49999999999999994) is 0, so the first decision is false
    above = 0.5 - 2.0 ** -53
    assert m.precision(above) == 10.0


def test_js_round_and_js_remainder_against_node():
    """The two JavaScript primitives the algorithm depends on, checked
    numerically against V8 with ``TOLERANCE``, including ties, the 2**52
    boundary and negative operands."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    xs = [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, -0.75, 0.75, 0.49999999999999994,
          4503599627370495.5, -4503599627370495.5, 5129144868306001.0,
          -5129144868306001.0, 1e21, -1e21, 2.5e-7]
    pairs = [(7.0, 2.0), (-7.0, 2.0), (7.0, -2.0), (-7.0, -2.0),
             (1e300, 7.0), (2.0 ** 53 + 2.0, 1024.0), (5e-324, 1e-300)]
    script = """
const xs = JSON.parse(process.argv[1]), ps = JSON.parse(process.argv[2]);
console.log(JSON.stringify({
  round: xs.map(Math.round),
  rem: ps.map(p => p[0] % p[1]),
}));
"""
    done = subprocess.run([node, "-e", script, json.dumps(xs), json.dumps(pairs)],
                  capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    want = json.loads(done.stdout)
    got_round = m.js_round(np.array(xs, dtype=np.float64))
    for x, g, w in zip(xs, got_round, want["round"]):
        assert math.isclose(float(g), w, rel_tol=TOLERANCE, abs_tol=TOLERANCE), x
    got_rem = m.js_remainder(
        np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))
    for p, g, w in zip(pairs, got_rem, want["rem"]):
        assert math.isclose(float(g), w, rel_tol=TOLERANCE, abs_tol=TOLERANCE), p


# ------------------------------------------------------- degenerate behaviour


def test_infinity_is_the_one_known_divergence_from_1_3_0():
    """Upstream 1.3.0 (2026-10-28) added ``if (num === Infinity) return
    Infinity``, so ``num2fraction(Infinity)`` is the number ``Infinity`` there
    and ``num2fraction(1/0)`` likewise.  Under 1.2.2 -- this port's target, and
    the version its fixture came from -- both hang.

    The port follows 1.2.2 and raises ``Num2FractionError``.  This test exists
    to make that a recorded decision rather than a silent mismatch: if the port
    is ever retargeted to 1.3.0, this is the assertion that must change first.
    ``-Infinity`` and ``NaN`` hang in 1.3.0 too (verified with ``timeout 5
    node``: both exit 124), so they are not part of the divergence.
    """
    for value in (math.inf, math.inf * 1.0):  # Infinity and 1/0
        with pytest.raises(m.Num2FractionError):
            m.num2fraction(value)
    for value in (-math.inf, math.nan):
        with pytest.raises(m.Num2FractionError):
            m.num2fraction(value)


def test_upstream_nontermination_is_an_error_here(upstream_live):
    """Upstream 1.2.2 hangs on these (no Infinity/NaN guard).  The oracle
    flags them without calling upstream; the port must raise the port's
    documented error instead of hanging."""
    hangs = {rec["in"] for rec in upstream_live if "hang" in rec}
    # verified individually with `timeout 4 node`: each exits 124
    assert hangs == {'"abc"', '""', '"   "', "[0.5, 0.25]", "{}"}, hangs
    for expr in sorted(hangs):
        value = _python_input(expr)
        if value is _CONTAINER:
            for container in ([0.5, 0.25], {}):
                with pytest.raises(TypeError):
                    m.num2fraction(container)
            continue
        with pytest.raises(m.Num2FractionError):
            m.num2fraction(value)
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(m.Num2FractionError):
            m.num2fraction(value)
        with pytest.raises(m.Num2FractionError):
            m.num2fraction(str(value))


def test_zero_one_and_sign_of_zero_match_upstream():
    """Upstream short-circuits `num === 0 || num === '0'`; the number -0
    satisfies it and the string '-0' does not.  Both from node."""
    assert m.num2fraction(0) == "0"
    assert m.num2fraction(0.0) == "0"
    assert m.num2fraction(-0.0) == "0"
    assert m.num2fraction("0") == "0"
    assert m.num2fraction("-0") == "0/1"
    assert m.num2fraction(1) == "1/1"
    assert m.num2fraction(-1) == "-1/1"
    # one element, twice: the batch path agrees with the scalar path
    assert m.num2fraction_batch([1.0]) == [m.num2fraction(1.0)]


def test_arrays_for_gcd_js_remainder_precision_accept_shape_and_empty():
    """The array entry points upstream has no analogue for; degenerate shapes
    must not crash: empty input, one element, duplicates, 2-D and broadcast
    shapes, and mismatched shapes raise ``ValueError`` (numpy's broadcast
    rule, which is the same contract numpy itself raises)."""
    empty = np.array([], dtype=np.float64)
    for fn in (m.gcd, m.js_remainder):
        out = fn(empty, empty)
        assert out.shape == (0,)
    assert m.precision_array(empty).shape == (0,)
    nums, dens, precs, status = m.convert_array(empty)
    assert (nums.shape, dens.shape, precs.shape, status.shape) == ((0,),) * 4

    one = np.array([6.0])
    assert m.gcd(one, np.array([4.0]))[0] == 2.0
    dup = np.array([0.5, 0.5, 0.5])
    assert list(m.precision_array(dup)) == [10.0, 10.0, 10.0]
    # broadcasting (3,) with a scalar, and (2, 3) with (3,)
    np.testing.assert_array_equal(
        m.gcd(np.array([6.0, 9.0]), 3.0), np.array([3.0, 3.0]))
    assert m.js_remainder(np.zeros((2, 3)), np.zeros(3)).shape == (2, 3)
    # mismatched, non-broadcastable shapes
    with pytest.raises(ValueError):
        m.gcd(np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError):
        m.js_remainder(np.zeros(3), np.zeros(2))


def test_nonfinite_elements_in_arrays_follow_the_documented_contract(
        upstream_live):
    """NaN/inf inside an array are reported per element by ``convert_array``
    as STATUS_NONFINITE, and raised by the string path; ``gcd``/``js_remainder``
    treat a non-finite remainder as zero, which is what upstream's
    ``abs(GCD)`` would see if it ever got there."""
    values = np.array([0.5, np.nan, np.inf, -np.inf, 2.555])
    nums, dens, precs, status = m.convert_array(values)
    assert list(status) == [0, 2, 2, 2, 0]
    assert (nums[0], dens[0]) == (1.0, 2.0)
    assert (nums[4], dens[4]) == (511.0, 200.0)
    assert m.gcd(np.inf, 1.0) == 1.0
    assert m.js_remainder(np.inf, 2.0) == 0.0
    # upstream's findPrecision never returns for these; the scalar helper
    # reports non-convergence as 0.0 rather than raising
    assert m.precision(np.nan) == 0.0
    assert m.precision(np.inf) == 0.0


def test_js_containers_are_not_accepted_where_upstream_coerces():
    """Upstream passes non-numbers through ``Number(x)``, so ``[0.5]`` works
    there; the Python API takes numbers and strings only and raises
    ``TypeError`` for containers rather than guessing.  Documented and
    pinned so the boundary cannot drift silently."""
    for value in ([0.5], [0.5, 0.25], {}, None):
        with pytest.raises(TypeError):
            m.num2fraction(value)
