# mojo-num2fraction

A Mojo port of [`num2fraction`](https://github.com/yisibl/num2fraction) — the
npm package that turns a number into a fraction string, `0.2` → `"1/5"`,
`2.555` → `"511/200"`. The whole algorithm is now a compiled kernel, and the
Python package is a ctypes shim over it.

```
PYTHONPATH=python python -c "import mojo_num2fraction as m; print(m.num2fraction(2.555))"
# 511/200
```

## What the upstream actually is

`num2fraction` 1.2.2 is 40 lines of JavaScript:

```js
function almostEq (a, b) { return abs(a - b) <= 9.5367432e-7 }
function GCD (a, b) { return almostEq(b, 0) ? a : GCD(b, a % b) }
function findPrecision (n) {
  var e = 1
  while (!almostEq(round(n * e) / e, n)) e *= 10
  return e
}
```

That is the entire package. There is no IO, no config, no CLI: four numeric
islands, three of which are data-dependent loops, plus a string concatenation.
So the port is not a subset here — it is the package, with the string
concatenation left in Python where the formatting decisions belong.

There is no `num2fraction` on PyPI; the name belongs to an npm package. The
Python package here is `mojo_num2fraction` and it imports alongside nothing
in particular.

## Coverage

| Upstream surface | Ported | Where |
| --- | --- | --- |
| `findPrecision(n)` — the power-of-ten search | yes | `n2f_precision`, `n2f_precision_array` |
| `GCD(a, b)` — Euclidean on doubles, JS remainder | yes | `n2f_gcd`, `n2f_gcd_array` |
| `num2fraction(num)` — numerator/denominator | yes | `n2f_convert_value`, `n2f_convert_range` |
| the `'0'` short-circuit for `0` and `'0'` | yes | `num2fraction` |
| `parseFloat` on string input (`'1.5px'` → 3/2) | yes, in Python | `parse_float` |
| `String(number)` formatting of the result | yes, in Python | `_js_number_to_string` |
| recursion → iteration in `GCD` | changed | iterative, same result |
| iteration cap in `findPrecision` | **added** | see below |

Nothing was left out. Three JavaScript behaviours had to be reproduced
exactly rather than approximately, and all three are covered by tests.

## The three things that are easy to get wrong

**`%` is `fmod`, not a floored modulo.** JavaScript's remainder is the
truncated quotient with the sign of the dividend. Mojo's `Float64 %` — and
Python's — take the sign of the *divisor*. The GCD's Euclidean path inherits
this exactly, so the port implements `fmod` by hand. It is also *exact*, as
IEEE-754 requires: the short form `x - trunc(x / y) * y` rounds the product
and puts **95 of the 1616** fixture answers one ulp out, e.g.
`num2fraction(41033158946.448006)` returns `5129144868306002/125000` instead of
`5129144868306001/125000`. The kernel uses `fma(q, y, -x)`, which rounds once
and is therefore exact, and falls back to a binary reduction (double the
divisor until it is within a factor of two of the remainder, subtract once —
exact by Sterbenz) for quotients at or above 2^53, including quotients that
overflow to infinity. Verified against `math.fmod` on 50 000 random pairs
spanning the whole exponent range, subnormals included: zero disagreements.

A "scale both operands down until the quotient fits" scheme cannot work here,
and the kernel comment says why: the quotient of two operands scaled by the
same power of two is unchanged, so the scaling never makes it representable.

**`Math.round` is not `floor(x + 0.5)` above 2^52.** There the ulp is 1 or
more, the addition rounds before the floor does, and the two disagree on
exactly the values that matter: V8 returns `5129144868306001`, while
`floor(5129144868306001 + 0.5)` is `...002`. Doubles at or above 2^52 are
already integral, so ECMA-262 returns them unchanged, and that identity branch
is what the kernel does.

**`String(number)` is the shortest round-trip, not the exact value.** The
numerator of `num2fraction(70142542814845430.0)` is the double
`70142542814845432`, and `String` prints `"70142542814845430"`. `str(int(x))`
prints the exact value and is wrong. The Python side implements the ECMA-262
formatting rules; checked against node on 5018 values (5000 of them random bit
patterns), zero disagreements.

## Deliberate differences

* **Non-finite inputs raise.** `Math.round(inf * e) / e` is `NaN`, which never
  compares within the epsilon, so upstream's `findPrecision` multiplies by ten
  forever. Verified: `timeout 3 node -e "require('num2fraction')(Infinity)"`
  exits 124. `Num2FractionError` is raised instead of hanging.
* **This is the one divergence from upstream 1.3.0.** Upstream 1.3.0
  (2026-10-28) added `if (num === Infinity) return Infinity`, so
  `num2fraction(Infinity)` is the number `Infinity` there; 1.3.0's own
  `test/test.js` asserts it. The port targets 1.2.2 -- the published version
  its fixture came from -- and raises for `Infinity` as it does for `NaN` and
  `-Infinity`, which 1.3.0 also hangs on (verified: `timeout 5 node` exits 124
  for both). `tests/test_parity_live.py::test_infinity_is_the_one_known_divergence_from_1_3_0`
  records this so it stays a decision rather than a surprise.
* **`findPrecision` is capped at 400 steps.** `e` overflows to infinity well
  before that, and a non-converging search is already caught by the
  finiteness check. A 300 000-value sweep of non-integral doubles between 1e4
  and 1e18 found no finite input that fails to converge, so this is a bound,
  not a behaviour change; it exists so the kernel can never spin.
* **`GCD` is iterative** and guarded at 128 steps, where upstream's recursion
  would exhaust the stack.
* The GCD's sign follows the algorithm, not the inputs, exactly as upstream:
  `gcd(-5, 10)` is `-5.0` and the caller takes `abs`.

## Not implemented

Nothing in the package is missing. There is no `main` export, no CLI and no
config in upstream to forward.

## Install and run

```bash
source /nvme0n1-disk/mojo-toolchain/activate.sh   # mojo 1.2.0.dev2026092605
bash build/build.sh                               # -> dist/libmojo-num2fraction.so
PYTHONPATH=python /nvme0n1-disk/mojo-toolchain/testvenv/bin/python -m pytest tests -q
```

`tests/conftest.py` skips the module if the shared library has not been built.

## Tests

35 tests, all against real behaviour — nothing here only checks that the shim
imports. Two files: `tests/test_num2fraction.py` (the original suite, unchanged)
and `tests/test_parity_live.py` (parity against upstream, added in the TEST
phase; see below).

* **1616 upstream values, exact.** `tests/upstream_vectors.json` was produced by
  `tests/upstream_vectors.js` running the real `index.js` under node, and the
  expected strings are its output. Regenerate with
  `node tests/upstream_vectors.js /path/to/num2fraction/index.js`. A separate
  pure-Python implementation of the same algorithm is checked against the same
  fixture first, so a shared misreading would have to be made twice.
* **fmod exactness** at `rtol=0, atol=0` against `math.fmod`, 4000 random
  full-range pairs plus a table of extreme ratios, and an explicit check that
  the floored modulo would have disagreed on many of them.
* **`Math.round` ties and the 2^52 boundary**, with node's answers.
* **The epsilon's effect** on the precision search: `precision(1e-7) == 1.0`
  and `precision(1e-6) == 1e6` are both decisions of that one constant.
* **Bit-pattern inputs** — 2000 doubles straight from random 64-bit patterns,
  so nothing decimal is assumed about the magnitudes.
* **A 64 MiB buffer whose address is above 4 GiB**, because a ctypes
  `argtypes` list one entry short is silent: ctypes zero-fills the missing
  slot, the address is truncated, and every small-array test still passes.
* The number `'0'` and the string `'-0'` take different paths upstream
  (`num === 0` is true for one and false for the other); both are asserted.

### Parity against a live upstream (test_parity_live.py)

Ground truth here is upstream's code run at test time, not hand-copied
constants.

* `tests/upstream_oracle.js` loads the real `index.js` and answers ~4 970
  generated inputs: 2 400 random doubles (long and short decimal expansions),
  the README's exact rationals, the doubles within an ulp of 0.5, ties and
  negatives across 13 decades, degenerate strings (`''`, `'abc'`, `'0x10'`,
  `'1/2'`, `.5`, `' 7 '`...) and JS containers. Set `NUM2FRACTION_UPSTREAM` to
  an `index.js`; otherwise a cached npm 1.2.2 tarball is used, fetched once if
  npm is present. Without node the live comparisons skip with a reason.
* **`test/` from upstream's repo is the second fixture.** Its `test.js` (the
  file `npm test` runs) is embedded verbatim, including the two `t.notEqual`
  rows.
* **Every public function is compared**: `num2fraction` (strings, exact),
  `convert` (numerator, denominator, precision, against upstream's string
  parsed back to numbers), `precision` (against upstream's `findPrecision`
  re-run in the oracle, since it is not exported), `gcd` (against upstream's
  `GCD` on the `(number, precision)` pairs upstream actually feeds it),
  `fraction`, `num2fraction_batch`, `convert_array`, `precision_array`,
  `js_round` and `js_remainder` (the last two against V8 directly).
* **Tolerance is written down.** `TOLERANCE = 2 * math.ulp(0.0)` — i.e. zero —
  applied with `math.isclose` to every numeric comparison. Measured, not
  assumed: the port is bit-identical to upstream on all ~4 970 live values, so
  any deviation at all is a parity failure rather than a rounding allowance.
* **Degenerate inputs upstream defines**: empty input, one element, duplicates,
  ±0, NaN/infinity, 2-D and broadcast shapes, non-broadcastable shapes (which
  raise `ValueError`). Where upstream hangs (non-numeric strings, `NaN`,
  `Infinity`, multi-element arrays, objects) the port raises
  `Num2FractionError`; where upstream coerces a container through `Number(x)`,
  the Python API raises `TypeError` instead of guessing.

Mutations that the suite catches, re-measured after the TEST phase by
rebuilding each: changing the epsilon constant to `1e-6` (11 failures); writing
`_js_round` as `floor(x + 0.5)` (2 failures —
`test_js_round_near_half_matches_v8` and the direct V8 comparison, because
`findPrecision` lands on the same `e` for the doubles where the two disagree);
always rounding up instead of comparing against the half (14 failures).
Replacing the `fma` with `ax - trunc(q) * ay` is *not* caught — because Mojo
contracts that expression back into the same FMA, so it was never a different
program.

One mutation that is *not* caught, stated so nobody reads more into the 35
green tests than is there: dropping `_js_round`'s "integral x is returned
unchanged" branch — keeping the half comparison for every input — still passes.
That is not an oversight in the tests: the branch-free form returns `x` for
every integral `x` as well (`x - floor(x)` is 0, so it does not round up), and
doubles at or above 2**52 are all integral, so the two forms compute the same
function on every finite input. The branch stays because it is the ECMA-262
statement of the rule and easy to check by eye; it is redundant, which is
exactly why no test can tell the difference.

The TEST phase found and fixed one real bug: `_js_round` used
`floor(x + 0.5)`, which is not `Math.round`. V8 gives
`Math.round(0.49999999999999994) === 0` while that sum rounds to 1, and
`Math.round(-0.4)` is `-0` while the shortcut gives `+0`. The kernel now
implements ECMA-262 with an explicit floor/ceil, and `MOJO_NOTES.md` §1 records
the toolchain trap that surfaced on the way: on this compiler `Float64`
comparisons order zeros by bit pattern, so `-0.4 < -0.0` is true and
`std.math.floor(-0.4)` is `-0.0`.

## GPU

**CPU only: memory-bound.** No GPU path exists, and none was written. The gate
was checked on this checkout rather than assumed; the evidence is in
`bench/gpu_gate.md`.

* The shared GPU was present and had **5 470 MiB** free, so the memory gate
  passed.
* Per-byte arithmetic is below the cut for every kernel but one. On the
  conservative count of `bench/analysis.md` (re-run this session), the best row
  is manydigits `convert_array` at **4.48 flops/byte**; `typical` is 1.14–1.99
  and `huge` is 0.75 (a 1 519-step dependent subtract chain, the worst possible
  GPU shape). A generous re-count of every op in `src/kernels.mojo` puts
  manydigits at 9.23, still under the break-even ratio for `Float64` on this
  card.
* Only manydigits clears 2 flops/byte, and its kernel is 21.5 sequential
  `_js_remainder` calls per element (measured), so its cost is dependency depth,
  not bandwidth. It also already runs at 55 ms for 1e6 elements.
* Hard blocker on top of that: this environment cannot compile a GPU kernel.
  `max.gpu.host` needs the `max` conda package (`MOJO_NOTES.md` §5b), and
  `pixi.toml` pins `mojo` only, so `from max.gpu import block_idx, thread_idx`
  fails with `unable to locate module 'max'`. Adding `max` would also move the
  toolchain pin: the channel's newest `max` is `26.8.0.dev2026111005`, not the
  lockstep `26.7.0.dev2026092905`.

Because there is no GPU code, there are no GPU tests to skip: the suite is
35 tests on the CPU path and needs no device. The CPU path is unchanged, so
before and after numbers are identical and the timings in Benchmarks below
stand. Sizes real callers pass are 1 to 1e6 elements and the GPU wins at none
of them, so there is nothing to revert.

## Benchmarks

Xeon E5-2697 v4, 36 cores, shared with other work, mojo 1.2.0.dev2026092605.
Min of 7 runs. The correctness gate runs first and compares the kernel against
the Python reference on 20 000 values per input class plus the 1616 upstream
values; if anything disagrees, no timings are printed.

There is no vectorised NumPy formulation of this algorithm to compare against
— the precision search and the GCD are per-element data-dependent loops — so
the fair baseline is upstream's own algorithm executed in Python.

| input class | n | Mojo batch | Python loop | speedup | scalar ctypes per element |
| --- | ---: | ---: | ---: | ---: | ---: |
| typical (0.001–64, 1–3 digits) | 1 000 | 0.17 ms | 4.9 ms | 29.5x | 6.50 ms |
| typical | 100 000 | 3.85 ms | 249.8 ms | 64.9x | 650.62 ms |
| typical | 1 000 000 | 11.88 ms | 2 087.8 ms | 175.7x | 7 353.40 ms |
| many digits | 1 000 | 0.33 ms | 6.0 ms | 18.3x | 6.59 ms |
| many digits | 100 000 | 10.55 ms | 696.1 ms | 66.0x | 678.05 ms |
| many digits | 1 000 000 | 53.43 ms | 6 425.1 ms | 120.3x | 7 079.69 ms |
| huge (\|x\| > 2^53) | 1 000 | 2.44 ms | 1.4 ms | **0.6x** | 8.58 ms |
| huge | 100 000 | 75.70 ms | 187.4 ms | 2.5x | 872.51 ms |
| huge | 1 000 000 | 310.35 ms | 1 572.2 ms | 5.1x | 8 667.25 ms |

Re-measured in the GPU phase after the unroll and SIMD phases had landed (same
machine, same correctness gate, green: `flock /tmp/mojo-bench.lock` then
`python bench/bench.py`, exit 0).

The `huge` row is a real result and reported as one: when `|x|` passes 2^53
the GCD's first division takes the binary-reduction path, which runs once per
bit of the quotient. At n = 1 000 the Python loop is **faster** than the batch
call (1.4 ms against 2.44 ms); the batch only pulls ahead as it grows.

Thread scaling, typical, 1e6 elements, chunked from the shim over a
`ThreadPoolExecutor` (the kernel takes a range, so a chunk carries no state):

| workers | 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| ms | 64.06 | 32.25 | 17.46 | 10.72 |
| ns/element | 64.1 | 32.3 | 17.5 | 10.7 |

6.0x on 8 workers on this run. This box is shared, so the figure moves with the
load on it.

## Layout

```
build/build.sh              src/kernels.mojo -> dist/libmojo-num2fraction.so
src/kernels.mojo           one compilation unit, 8 exports
python/mojo_num2fraction/  _lib.py (ctypes), core.py (API), __init__.py
tests/                      test_num2fraction.py, upstream_vectors.{js,json}
bench/bench.py              correctness-gated timing
bench/analysis.md           where the time goes (ANALYZE phase)
bench/gpu_gate.md           GPU gate evidence (this phase): why CPU only
```

Licence: MIT, as upstream.
