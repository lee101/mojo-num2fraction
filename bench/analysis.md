# ANALYZE: where the time goes in mojo-num2fraction

Measured, not estimated. Every number below came from a run in this session
(scripts in `/tmp/n2f_analysis/`). Nothing in `src/` or `tests/` was changed;
this file is the only addition. Rebuilding `src/kernels.mojo` produced a `.so`
byte-identical to the shipped one, so all numbers describe the code in the repo.

Machine: Intel Xeon E5-2697 v4 (AVX2 + FMA, **no AVX-512**), 72 hardware
threads = 36 cores, AVX2 = 4 doubles/vector, 2.3 GHz base. Shared box, load
average 26–47 during the runs. mojo 1.2.0.dev2026092905, Python 3.13.14,
numpy 2.5.1, node v22.22.2.

## What was run

| run | how | what it gave |
| --- | --- | --- |
| project bench | `pixi run bench` (flock-wrapped) | exit 0; correctness gate green (1616 upstream values + 20 000/class vs the Python reference) |
| entry-point timing | `probe.py`, `probe2.py`, `glue.py`, `glue2.py`, all under `flock /tmp/mojo-bench.lock` | min of 5–7 runs; per-element loops on 1 000 elements after warm-up |
| upstream comparison | real `num2fraction@1.2.2` from `npm pack`, node v22, one process per (class, n), `process.hrtime` around each call | per-call cost at equal n |
| parity sanity | `parity.py`: port vs node strings on 3 × 10 000 values | **0 mismatches**, so the comparison is like-for-like |
| kernel vs glue | `glue2.py`: `convert_array` vs the *same* 4/8-way kernel call with preallocated buffers | the split in §1 |
| dynamic counts | Python mirror of `src/kernels.mojo`, 10 000 values/class | iteration counts behind the flops column |
| codegen | `objdump -d dist/libmojo-num2fraction.so` | the facts in §5 |

## Headline

1. **For `num2fraction` / `num2fraction_batch`, the kernel is 0.5–17 % of the
   time.** 98–99.5 % of the typical-value batch is Python: `tolist` + a
   per-element loop that formats two doubles. That loop, not the arithmetic, is
   the biggest lever in the package.
2. **For `convert_array`, the kernel is ~95–99 %** (97.9–98.6 % typical,
   97.1 % manydigits, 94.7 % huge at n = 1e5; 77.9 % typical at n = 1e6, where
   thread-pool dispatch costs 2.9 ns/element). The kernel is the whole story
   for this API.
3. **The `huge` class costs 20–70× more per element than typical**: 764 ns vs
   37 ns at n = 1e5. The GCD's binary-reduction path runs once per bit of the
   quotient — **1 519 subtract steps per element** at 10 000 values. That is
   also the class where the port loses to upstream (§2).
4. **No row qualifies for GPU.** Max measured flops/byte is **4.48**
   (manydigits); the GPU threshold here is ≈ 20–40. The only row that would
   clear it on explicit ops (huge, ≈ 75 flops/byte once the 1 519 serial
   subtracts are counted) is a dependent subtract chain, not throughput.

## 1. Ranked hotspot table

Time is per call for batch APIs. `bytes` is compulsory traffic: input read +
outputs written per API call, streaming (L1/L2 reuse not credited). `flops`
are counted from the Python mirror (`work.py`), conservative scalar count
including compares; `flops/byte` divides the two.

| # | function | class | size | time | bytes | flops | flops/byte | bound | time sits in |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `num2fraction_batch` | huge | 1e6 | 4 370 ms | 28.0 MB | 2.70e7 (+1.52e9 subtract steps) | 0.75 † | serial dep-chain | **python** 91 %, kernel 9 % |
| 2 | `num2fraction_batch` | manydigits | 1e6 | 2 949 ms | 28.0 MB | 1.61e8 | 4.48 | latency (branchy) | **python** 98 % |
| 3 | `num2fraction_batch` | typical | 1e6 | 2 424 ms | 28.0 MB | 7.17e7 | 1.99 | latency (branchy) | **python** 99.5 % |
| 4 | `js_remainder(array, 1)` | huge | 1e6 | 2 434 ms | 24.0 MB | 5.0e6 (+1.52e9 subtract steps) | 0.21 | serial dep-chain | kernel 100 % |
| 5 | `gcd(array, 1)` | huge | 1e6 | 2 426 ms | 24.0 MB | 5.0e6 (+1.52e9 subtract steps) | 0.21 | serial dep-chain | kernel 100 % |
| 6 | `num2fraction_batch` | huge | 1e5 | 443 ms | 2.8 MB | 2.70e6 | 0.75 † | serial dep-chain | python 83 %, kernel 17 % |
| 7 | `convert_array` | huge | 1e6 | **376 ms** | 28.0 MB | 2.70e7 (+1.52e8 subtract steps) | 0.75 † | serial dep-chain | kernel 100 % |
| 8 | `num2fraction_batch` | manydigits | 1e5 | 281 ms | 2.8 MB | 1.61e7 | 4.48 | latency (branchy) | python 96 % |
| 9 | `num2fraction_batch` | typical | 1e5 | 236 ms | 2.8 MB | 7.17e6 | 1.99 | latency (branchy) | python 98 % |
| 10 | `gcd(array, 1)` | manydigits | 1e6 | 144 ms | 24.0 MB | 1.05e8 | 4.36 | compute | kernel 100 % |
| 11 | `convert_array` | manydigits | 1e6 | 55.1 ms | 28.0 MB | 1.61e8 | 4.48 | compute | kernel 97 % |
| 12 | `gcd(array, 1)` | typical | 1e6 | 46.6 ms | 24.0 MB | 3.23e7 | 1.34 | compute | kernel 100 % |
| 13 | `convert_array` | huge | 1e5 | 76.4 ms | 2.8 MB | 2.70e6 (+1.52e7 subtract steps) | 0.75 † | serial dep-chain | kernel 94.7 % |
| 14 | `convert_array` | typical | 1e6 | 12.8–13.0 ms | 28.0 MB | 7.17e7 | 1.99 | mem | kernel **77.9 %** (dispatch 2.9 ns/elem) |
| 15 | `convert_array` | manydigits | 1e5 | 9.9–10.3 ms | 2.8 MB | 1.61e7 | 4.48 | compute | kernel 97.1 % |
| 16 | `js_round(array)` | any | 1e6 | 4.3–5.2 ms | 16.0 MB | 6.0e6 | 0.375 | mem | kernel 100 % |
| 17 | `convert_array` | typical | 1e5 | 3.5 ms | 2.8 MB | 7.17e6 | 1.99 | mem | kernel 98.6 % |
| 18 | `num2fraction` (scalar, 1 000-element loop) | typical | 1 | **9.01 µs** | 32 B | 56 | 1.7 | latency | **python 73 %**, ctypes 27 % |
| 19 | `convert` (scalar) | typical | 1 | 6.08 µs | 32 B | 56 | 1.7 | latency | ctypes 89 % (4.3 µs floor) |
| 20 | `precision` (scalar) | typical | 1 | 0.82 µs | 8 B | 36 | 4.5 | latency | ctypes 89 % (0.73 µs floor) |

† huge flops/byte counts explicit division/fma/compare only. With the
binary-reduction subtract steps included it is ≈ 75 flops/byte, but each of
those depends on the previous one (1 519-step chain), so the GPU verdict is
still no — see §4.

### Inside the kernel, as a share of `convert_array` at n = 1e5

`convert_array` runs its kernel 4-way at this n. These parts are separate
kernel entry points, so the two that differ in vectorisation are flagged, not
subtracted.

| hotspot | typical (37.5 ns/elem) | manydigits (99.3) | huge (764.3) |
| --- | ---: | ---: | ---: |
| `_find_precision` (`precision_array`, whole kernel) | **49.0 %** (18.4 ns) | **48.2 %** (47.9) | 0.6 % |
| `_js_round` (`js_round_array`, whole kernel) | 11.6 % (4.3) | 5.0 % (4.9) | 0.5 % |
| `_js_remainder` whole kernel | 14.1 % (5.3) | 5.6 % (5.5) | 316 % ‡ |
| `gcd` whole kernel | 186 % ‡ | 144 % ‡ | 319 % ‡ |
| Python + ctypes + alloc + dispatch (same 8-way path) | **1.4 %** (0.5 ns) | −7.3 % ‡ | 5.3 % (36.7 ns) |

‡ >100 % because `n2f_precision_array` / `n2f_js_remainder_array` / `n2f_gcd_array`
are **scalar** loops while `convert_array` is 4-way at n = 1e5: they cannot be
subtracted. The trustworthy decomposition is the non-negative one:
`_find_precision` 49 %, `_js_round` inside it 11.6 %, `_js_remainder` 14 % on
typical — and for `huge` the whole kernel reading (764 ns/elem), where the GCD
is the only work (precision costs nothing because every huge value is already
integral).

## 2. Upstream comparison (real npm 1.2.2, node v22)

Per scalar `num2fraction` call. Port = `num2fraction(float(x))` in a 1 000-element
Python loop. Node's per-call cost falls with n as V8 warms up (43.5 µs at
n = 1 → 2.06 µs at n = 100), so the large-n rows are the fair ones.

| caller size | class | port / call | upstream / call | ratio | |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | typical | 9.0 µs | 43.5 µs | 0.21× | port faster (node warming) |
| 1 | manydigits | 9.4 µs | 48.2 µs | 0.20× | port faster (node warming) |
| 1 | huge | 16.3 µs | 47.2 µs | 0.35× | port faster (node warming) |
| 100 | typical | 1.05 µs | 2.06 µs | 0.51× | port faster |
| 100 | manydigits | 0.92 µs | 4.91 µs | 0.19× | port faster |
| 100 | **huge** | 1.69 µs | 1.21 µs | **1.40×** | **port SLOWER** |
| 1 000 | typical | 10.4 µs | 1.49 µs | **7.0×** | **port SLOWER** |
| 1 000 | manydigits | 9.9 µs | 1.51 µs | **6.6×** | **port SLOWER** |
| 1 000 | **huge** | 13.5 µs | 0.85 µs | **15.9×** | **port SLOWER** |
| 10 000 | typical | 10.4 µs | 0.76 µs | **13.6×** | **port SLOWER** |
| 10 000 | manydigits | 9.9 µs | 1.40 µs | **7.1×** | **port SLOWER** |
| 10 000 | **huge** | 13.5 µs | 0.72 µs | **18.7×** | **port SLOWER** |
| 100 000 | typical | 2.36 µs | 0.29 µs | **8.2×** | **port SLOWER** |
| 100 000 | manydigits | 2.81 µs | 1.03 µs | **2.7×** | **port SLOWER** |
| 100 000 | **huge** | 4.43 µs | 0.44 µs | **10.0×** | **port SLOWER** |

Every "SLOWER" row is the Python tail, not the numeric kernel. At n = 100 000
the port's `convert_array` (numbers only) costs 37.5 / 99.3 / 764 ns per
element, while the full string call costs 2 361 / 2 814 / 4 429 ns per
element. Upstream's string call costs 290 / 1 028 / 444 ns per element. So
the port's numeric half is well ahead of V8's whole call on typical and
manydigits, about 1.7× *behind* on huge, and the Python string tail then costs
more than V8's entire call on every class.

Python baseline (from `pixi run bench`, README's own reference loop):
typical 1e6 2 086 ms vs port batch 12.1 ms (**172×**), manydigits 1e6
7 551 ms vs 50.1 ms (**151×**), huge 1e6 1 961 ms vs 319 ms (**6.1×**).

## 3. Python-level loops re-entering ctypes, and per-call copies

ctypes call floor measured on this box: **732 ns** for a 1-`c_double` call,
**4.3 µs** for the 3-argument `n2f_convert_value`.

| location | ctypes calls / element | measured |
| --- | ---: | ---: |
| `convert` (scalar) | 1 (3 args) + two `np.empty` | 6.08 µs |
| `num2fraction` (scalar) | 1 via `convert` | 9.01 µs, of which 1.9 µs is `_js_number_to_string` ×2 and ~1 µs dispatch/f-string |
| `precision` (scalar) | 1 (1 arg) | 0.82 µs (89 % floor) |
| `gcd(a, b)` scalar | 1 (2 args) | ~0.9 µs |
| `js_remainder(x, y)` scalar | 1 + `np.empty(1)` + 2 × `as_f64` | **10.5 µs** vs 5.5 **ns**/element on the array path — 1 900× |
| `js_round(x)` python float | 0 | 0.06 µs |
| `num2fraction_batch` | 0 ctypes, but 3 Python passes per element (`tolist`, zip loop, format) | 2.4 µs/element at 1e6 |

**Per-call allocations and copies:**

| site | cost (n = 1e6 unless stated) | note |
| --- | ---: | --- |
| `convert_array`: four `np.empty` | 27.9 µs (0.03 ns/elem) | negligible at 1e6; **2.3 µs of the 20.7 µs 1-element call** |
| `convert_array`: `ascontiguousarray(a.reshape(-1))` | 0.4 µs when already contiguous | full copy only for non-contiguous input |
| `num2fraction_batch`: three `.tolist()` | 124.6 ms | 5 % of the batch |
| `num2fraction_batch`: `zip` loop, no formatting | 151.1 ms | 6 % |
| `num2fraction_batch`: `_js_number_to_string` pairs | **1 734.7 ms** | **72 %** |
| `gcd(array, 1)` / `js_remainder(array, 1)`: `ascontiguousarray(broadcast_to(b, shape))` | one full 8 MB array | scalar `b` materialised to the whole shape every call |
| `js_remainder(scalar)`: `np.empty(1)` + 2 × `as_f64` | ~2 µs | dominates its 10.5 µs |

## 4. Next phase per hotspot

GPU is named only where the flops/byte figure would justify it — nowhere here.
Max measured is 4.48 (manydigits `convert_array`); the huge row at ≈ 75
flops/byte (subtracts counted) is a 1 519-step dependent chain, not bandwidth
work, and would keep its dependency depth on a GPU at worse launch cost.

| hotspot | figure | next phase |
| --- | --- | --- |
| `_js_number_to_string` loop in `num2fraction_batch` | 1 735 of 2 424 ms typical/1e6; 1.9 µs per pair; 73 % of the scalar call | **python** first (GIL-bound loop: `parallel` cannot help until it leaves Python), then **simd** to emit digits from the kernel so the tail disappears |
| `_find_precision` (typical, manydigits) | 49.0 % / 48.2 % of `convert_array`; 2.9 / 5.7 steps per element | **unroll** the search step (the compiler already vectorises its `x != t` as `vaddpd` + `vcmpltsd` + `vblendvpd`), **none** for wide SIMD (data-dependent trip count, no gather/scatter form) |
| `_gcd` / `_js_remainder` slow path (huge) | 1 519 subtract steps/element; 764 ns/element whole kernel; 0.21 explicit flops/byte | **unroll**: replace the doubling loop with a bit-length-derived exponent step, keeping Sterbenz exactness; then **parallel** (already 6.2–8.1× at 8 workers) |
| `_js_remainder` fast path | 14.1 % / 5.6 % of `convert_array`; one `vdivsd` per element | **unroll** (4-wide AVX2 is the ceiling here; `vfmadd*` is 0× in the disassembly today, so the FMA is being emitted as separate ops) |
| `js_round_array` | 4.3–5.2 ns/element; `vroundsd` + 1 compare + `vblendvpd`, 0 branches | **none** — latency-bound on one compare; no loop body left to unroll |
| `convert_array` glue | 1.4 % at 1e5, 22.1 % at 1e6 typical (thread-pool dispatch, not allocation) | **none** — this is the correct shape; the scalar API covers n = 1 |
| scalar `convert` / `precision` / `js_round` / `js_remainder` | 89 % of 0.82–10.5 µs is ctypes | **none** — a binding, already 0.73 µs from its floor |
| `gcd(array, 1)` / `js_remainder(array, 1)` | 2.4 s at huge/1e6; shim chunks only `convert_array` | **parallel** — same ranged split, no shared state |

## 5. Codegen facts for the next phase

- Rebuild is byte-identical to the shipped `.so`. No outgoing calls anywhere
  in the library (no libm `sqrt`/`fmod`).
- `vdivsd` 9×, `vsqrtsd` 0×, `vfmadd*` **0×**, `vmulpd` 0×, `vaddpd` 6×,
  `vroundsd` 10×, `vucomisd` 80×.
- `_js_round` is branch-free (`vroundsd` + `vcmplesd` + `vblendvpd`).
- `_find_precision`'s `x != t` is `vaddsd 1.0` + `vcmpltsd` + `vblendvpd`.
- Kernel speedup from 1 → 8 workers (same call path): typical 2.87× (1e5),
  5.88× (1e6); manydigits 4.76×, 5.92×; huge 6.19×, 8.09×. There is real
  parallel headroom before any SIMD work.

## Caveats

- Shared box, load average 26–47. Run-to-run spread ≤ 10 % on `convert_array`,
  up to 20 % on huge rows; two `glue2.py` rows came out ±3–7 % over 100 %
  (kernel ≥ full) for that reason — the splits are quoted with both runs' values
  where that happened.
- **`/tmp/mojo-bench.lock` is held by other sessions on this machine.** Several
  `flock … pixi run bench` holders were queued for the whole session; one of my
  calls waited 7.6 min without ever starting. Every timing above did acquire
  the lock; the parity check (correctness only) and disassembly did not.
- Upstream rows at n = 1 are unfair to node (mid-warm-up); those are marked.
- `flops` is a conservative scalar count from the Python mirror, not hardware
  micro-ops. It exists to bound the GPU verdict, and the 4.48× margin to the
  threshold exceeds a 2× counting error.
- The `gcd_array` / `js_remainder_array` / `precision_array` rows at 1e5 are
  scalar loops against a 4-way `convert_array`, hence §1's ‡ caveat. They are
  kept as rank rows (the rule is not to drop rows), not used as percentages.
- `pixi run bench` prints huge 1e6 at 319 ms against 376 ms here; that is the
  box's spread, not a code path difference.
