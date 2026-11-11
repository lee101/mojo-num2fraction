# GPU gate check (phase gpu)

Verdict: **CPU only: memory-bound.** No GPU path was written. This file records
the evidence, measured in this session on this checkout, so the decision can be
re-checked rather than trusted.

Machine at the time of the check:

| item | value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, PCIe gen 3 x16 |
| GPU memory free | **5 470 MiB** (`nvidia-smi --query-gpu=memory.free`) — above the 4 000 MiB floor |
| Toolchain | mojo 1.2.0.dev2026092905, from `pixi run` in this repo |

So the GPU was available and the free-memory gate passed. The phase stops on
the flops-per-byte gate instead, which is the first gate in the brief.

## Gate 1: flops per byte

Brief: only kernels above roughly **2 flops/byte** are candidates; below that,
write "CPU only: memory-bound" and stop.

Two independent counts were run against the actual kernel in `src/kernels.mojo`.
Both are lower bounds on arithmetic per byte, because non-numeric per-element
work (array indexing, control flow, status writes) is not counted and the output
bytes are counted at their full size.

### (a) Re-run of the count that produced `bench/analysis.md`

`/tmp/n2f_gate/flops_repro.py` re-implements the same accounting as the ANALYZE
phase's `work.py` and reproduces its column. Bytes are per API call: for
`convert_array`, 8 B in + 3 x f64 + i32 out = 36 B/element. `work.py` used a
28 B denominator for `convert_array`; both are shown.

| class | n | flops/byte (analysis.md denominator) | flops/byte (36 B: convert_array) |
| --- | ---: | ---: | ---: |
| typical | 1e6 | 1.46 | 1.14 |
| manydigits | 1e6 | 5.76 | **4.48** |
| huge | 1e6 | 0.96 | 0.75 |

The 4.48 reproduces the figure in `bench/analysis.md` exactly. Only
**manydigits** is above 2, and only at 2.2x the threshold.

### (b) Generous count of every arithmetic op in the kernel

`/tmp/n2f_gate/flops_gate.py` counts every op in `_find_precision`, `_js_round`,
`_js_remainder`, `_gcd` and `_convert_one` at its widest (each compare, select,
trunc, abs and the loop test counted as one), including the work that (a)
omits. Same inputs, same 36 B denominator:

| class | ops/element | flops/byte |
| --- | ---: | ---: |
| typical | 73.6 | 2.05 |
| manydigits | 332.2 | **9.23** |
| huge | 44.0 | 1.22 |

On this count manydigits is 4.6x the threshold. Either way, **the only candidate
is the manydigits class**, and `huge` and `typical` are excluded: `huge` is a
serial subtract chain (measured in `bench/analysis.md` at 1 519 dependent steps
per element), which is the worst possible shape for a GPU, not bandwidth work.

### The usual GPU rule of thumb, applied to this GPU

On a
data-centre card the flop:bandwidth ratio is much larger, so 9 ops/byte could in
principle be work for one. On the card in this box (RTX 5090, boost clock
reported at 3 105 MHz) the memory side is roughly 1.8 TB/s against roughly
100 TFLOP/s of **fp32**; this kernel is all `Float64`, and consumer cards run
fp64 at a small fraction of fp32, so the break-even ratio is far above 9. The
figure is quoted as an order of magnitude, not as a measured throughput.

## Gate 2: the toolchain cannot compile a GPU path here

This is independent of the numbers and it is a hard blocker.

`MOJO_NOTES.md` §5b: the `max.*` modules ship in the `max` conda package, not
in `mojo`. This repo's `pixi.toml` pins only `mojo = "==1.2.0.dev2026092905"`,
and `.pixi/envs/default/lib/python3.13/site-packages/../mojo` contains
`std.mojoc` and no `max.mojoc`. A two-line probe:

```mojo
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext

def main() raises:
    print("imports ok")
```

```
/tmp/n2f_gate/probe/p.mojo:1:6: error: unable to locate module 'max'
from max.gpu import block_idx, thread_idx
     ^
/tmp/n2f_gate/probe/p.mojo:2:6: error: unable to locate module 'max'
from max.gpu.host import DeviceContext
     ^
error: failed to parse the provided Mojo source module
```

`pixi run mojo build` exits 1. The channel has `max`, but the newest build there
is `26.8.0.dev2026111005`, not the `26.7.0.dev2026092905` that `MOJO_NOTES.md`
§5b says is released in lockstep with the pinned `mojo`. Adding `max` would move
the toolchain pin for a kernel that fails Gate 1.

## What is left undone and why

- No GPU kernel was written: Gate 1 excludes every kernel except manydigits,
  and manydigits' kernel is 21 sequential `_js_remainder` calls per element with
  a data-dependent trip count (measured: 21.46 remainder calls/element at
  1e6). One launch could cover it, but the per-element serial depth is what
  bounds it, and at 1e6 elements the transfer is 36 MB — a single
  copy-in/copy-out that a 5090 moves in ~0.1 ms, to replace a CPU kernel that
  already runs at 55 ms (`bench/analysis.md` row 11, 4.48 flops/byte).
- No GPU tests were added, so none are skipped: there is no GPU code path to
  test. The CPU path is unchanged and remains the only path. `pixi run build &&
  pixi run test` is green (35 passed).
- The README wording required by the brief is in place: "CPU only:
  memory-bound", with the before/after numbers being the same numbers, because
  no code changed.
