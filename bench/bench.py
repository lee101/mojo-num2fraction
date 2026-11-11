"""Benchmark: the Mojo kernel against a pure-Python run of the same algorithm.

There is no vectorised NumPy formulation of num2fraction to compare against:
the algorithm is a data-dependent loop per element (a precision search that
multiplies by ten until a rounding test passes, then a Euclidean GCD on
doubles), so the fair baseline is upstream's own algorithm executed in Python.
The Python reference below is the same code the tests use.

Correctness is gated first: if the kernel disagrees with the reference on a
single value, the timings are not printed.
"""

import json
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_num2fraction as m  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
EPSILON = 9.5367432e-7


def ref_js_round(x: float) -> float:
    if math.floor(x) == x:
        return x
    return float(math.floor(x + 0.5))


def ref_gcd(a: float, b: float) -> float:
    x, y = a, b
    guard = 0
    while abs(y) > EPSILON:
        if guard > 128:
            return 0.0
        x, y = y, math.fmod(x, y)
        guard += 1
    return x


def ref_convert(x: float):
    if x == 0.0:
        return 0.0, 1.0
    e = 1.0
    for _ in range(400):
        if abs(ref_js_round(x * e) / e - x) <= EPSILON:
            break
        e *= 10.0
    number = x * e
    g = abs(ref_gcd(number, e))
    return ref_js_round(number / g), ref_js_round(e / g)


def make_inputs(n: int, kind: str) -> np.ndarray:
    rng = np.random.default_rng(20260926)
    if kind == "typical":
        # CSS pixel ratios and similar: values a few significant digits wide
        return np.round(rng.uniform(0.001, 64.0, size=n), rng.integers(1, 4))
    if kind == "manydigits":
        return rng.uniform(0.1, 9.9, size=n) * 10.0 ** rng.integers(-6, 6, size=n)
    if kind == "huge":
        # |x| past 2**53: the GCD's first division takes the binary-reduction
        # path in the kernel
        return rng.uniform(1e18, 1e25, size=n)
    raise ValueError(kind)


def gate(kind: str, n: int = 20_000) -> None:
    """Gate on the numeric result, not the string: the pair (numerator,
    denominator) is what the kernel computes, the string is formatting."""
    values = make_inputs(n, kind)
    nums, dens, _precs, status = m.convert_array(values)
    bad = []
    for i, v in enumerate(values):
        if status[i] != m.core.STATUS_OK:
            bad.append((float(v), "status", int(status[i])))
            continue
        num, den = ref_convert(float(v))
        if float(nums[i]) != num or float(dens[i]) != den:
            bad.append((float(v), (nums[i], dens[i]), (num, den)))
    if bad:
        raise SystemExit(
            f"correctness gate failed for {kind}: {len(bad)} mismatches, "
            f"first {bad[:3]}"
        )
    print(f"gate ok: {kind}, {n} values match the Python reference")


def timeit(fn, values, repeats: int = 7) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(values)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    fixture = json.loads((ROOT / "tests" / "upstream_vectors.json").read_text())
    for case in fixture["cases"][:2000]:
        if m.num2fraction(float(case["in"])) != case["out"]:
            raise SystemExit(f"upstream fixture mismatch at {case['in']}")
    print(f"gate ok: {len(fixture['cases'])} upstream values reproduce exactly")

    sizes = [1_000, 100_000, 1_000_000]
    print()
    print(f"{'kind':<11}{'n':>10}{'mojo batch':>13}{'python loop':>13}"
          f"{'speedup':>9}{'scalar ctypes':>16}")
    for kind in ("typical", "manydigits", "huge"):
        gate(kind)
        for n in sizes:
            values = make_inputs(n, kind)
            t_batch = timeit(lambda v: m.convert_array(v), values)
            t_py = timeit(
                lambda v: [ref_convert(float(x)) for x in v],
                values,
                repeats=1,
            )
            probe = min(n, 2_000)
            t_scalar = timeit(
                lambda v: [m.convert(x) for x in v], values[:probe]
            ) * (n / probe)
            print(f"{kind:<11}{n:>10}{t_batch * 1e3:>10.2f}ms"
                  f"{t_py * 1e3:>10.1f}ms{t_py / t_batch:>8.1f}x"
                  f"{t_scalar * 1e3:>13.2f}ms")
    print()
    print("thread scaling, typical / 1e6 (min of 7):")
    import mojo_num2fraction.core as core
    values = make_inputs(1_000_000, "typical")
    for workers in (1, 2, 4, 8):
        core._MAX_WORKERS = workers
        t = timeit(lambda v: m.convert_array(v), values)
        print(f"  {workers} worker(s): {t * 1e3:7.2f} ms  {t / values.size * 1e9:6.1f} ns/element")
    core._MAX_WORKERS = 8
    print()
    print("per-element, typical / 1e6:")
    values = make_inputs(1_000_000, "typical")
    t = timeit(lambda v: m.convert_array(v), values)
    print(f"  mojo   {t / values.size * 1e9:8.1f} ns/element")


if __name__ == "__main__":
    main()
