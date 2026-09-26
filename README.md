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

18 tests, all against real behaviour — nothing here only checks that the shim
imports.

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

Mutations that the suite catches, checked by rebuilding: changing the epsilon
constant (4 failures), and dropping the 2^52 identity branch from `Math.round`
(5 failures). Replacing the `fma` with `ax - trunc(q) * ay` is *not* caught —
because Mojo contracts that expression back into the same FMA, so it was never
a different program.

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
| typical (0.001–64, 1–3 digits) | 1 000 | 0.15 ms | 12.3 ms | 80x | 16.0 us |
| typical | 100 000 | 7.1 ms | 761 ms | 107x | 14.5 us |
| typical | 1 000 000 | 30.2 ms | 4 826 ms | 160x | 11.3 us |
| many digits | 1 000 000 | 127.7 ms | 17 083 ms | 134x | 12.7 us |
| huge (\|x\| > 2^53) | 1 000 | 2.6 ms | 2.9 ms | **1.1x (parity)** | 17.8 us |
| huge | 1 000 000 | 544.7 ms | 4 196 ms | 7.7x | 14.8 us |

The `huge` row is a real result and reported as one: when `|x|` passes 2^53
the GCD's first division takes the binary-reduction path, which runs once per
bit of the quotient, and at n = 1000 the kernel only matches the Python loop.
It pulls ahead as the batch grows.

Thread scaling, typical, 1e6 elements, chunked from the shim over a
`ThreadPoolExecutor` (the kernel takes a range, so a chunk carries no state):

| workers | 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| ms | 61.0 | 47.1 | 33.1 | 32.4 |
| ns/element | 61.0 | 47.1 | 33.1 | 32.4 |

1.9x on 8 workers rather than the 3.4x the toolchain reference records — this
box is shared and was running other builds during the measurement.

## Layout

```
build/build.sh              src/kernels.mojo -> dist/libmojo-num2fraction.so
src/kernels.mojo           one compilation unit, 8 exports
python/mojo_num2fraction/  _lib.py (ctypes), core.py (API), __init__.py
tests/                      test_num2fraction.py, upstream_vectors.{js,json}
bench/bench.py              correctness-gated timing
```

Licence: MIT, as upstream.
