// Live oracle: runs the REAL upstream module and prints one JSON line per input.
//
//   node tests/upstream_oracle.js /path/to/num2fraction/index.js
//
// The corpus (random doubles, exact rationals, ties, degenerate strings and
// containers) is generated here, deterministically.
//
// Each line is {"in": <the JS expression used>, "out": <result>} or
// {"in": ..., "throws": "<Error class>", "message": "..."} or
// {"in": ..., "hang": true} when upstream's findPrecision would never
// terminate (it has no iteration cap, so node spins until killed -- the probe
// uses the same convergence pre-check the port's kernel uses, which is
// `findPrecision` from upstream's own index.js).
//
// The test suite feeds this script random values and the degenerate cases, so
// the expectations in test_parity_live.py are upstream's answers, produced at
// test time by upstream's code, rather than hand-copied constants.

'use strict'

const n2f = require(process.argv[2])

const EPSILON = 9.5367432e-7 // upstream index.js: almostEq
const MAX_STEPS = 400 // matches src/kernels.mojo MAX_PRECISION_STEPS

// upstream index.js: findPrecision, unrolled so a non-converging value does
// not wedge the probe.  This is a *detector*, not the expected value: the
// expected value below always comes from calling the real n2f().
function terminates (n) {
  let e = 1
  for (let i = 0; i < MAX_STEPS; i++) {
    if (Math.abs(Math.round(n * e) / e - n) <= EPSILON) return true
    e *= 10
    if (!isFinite(e)) return false
  }
  return false
}

// BigInt LCG so the generated doubles are deterministic across node versions.
class Rng {
  constructor (seed) {
    this.state = BigInt(seed) & ((1n << 64n) - 1n)
  }

  next () {
    const M64 = (1n << 64n) - 1n
    this.state = (this.state * 6364136223846793005n + 1442695040888963407n) & M64
    return Number(this.state >> 11n) / 9007199254740992
  }

  // a uniform double with a long decimal expansion (long denominators)
  longDecimal () {
    return (this.next() * 9000 + 1000) * (this.next() < 0.5 ? 1 : -1)
  }

  // a short decimal, the way CSS authors write ratios
  shortDecimal () {
    const digits = 1 + Math.floor(this.next() * 6)
    const mag = Math.floor(this.next() * 7) - 3
    const s = (this.next() * 9 + 1).toFixed(digits)
    const x = parseFloat(s) * Math.pow(10, mag)
    return this.next() < 0.5 ? -x : x
  }
}

// Runs one JS expression through upstream.  `value` is what upstream ends up
// parsing it to -- parseFloat for strings, Number(x) for anything else -- since
// upstream calls findPrecision on exactly that.
function probe (label, expr, value) {
  if (typeof value === 'number' && !terminates(value)) {
    return { in: label, hang: true }
  }
  try {
    const out = n2f(eval(expr)) // eslint-disable-line no-eval
    // JSON has no Infinity; upstream 1.2.2 hangs there anyway.
    const rec = { in: label, out: typeof out === 'number' && !isFinite(out) ? String(out) : out }
    // upstream's findPrecision itself, unrolled only for the step cap: the
    // precision search is not exported, so this is how the tests see it.
    if (typeof value === 'number' && value === value) {
      let e = 1
      for (let i = 0; i < MAX_STEPS; i++) {
        if (Math.abs(Math.round(value * e) / e - value) <= EPSILON) break
        e *= 10
      }
      rec.precision = isFinite(e) ? e : null
      // the first decision of that loop, which is where Math.round shows
      rec.first_step = Math.abs(Math.round(value) - value) <= EPSILON
    }
    return rec
  } catch (err) {
    return { in: label, throws: err.constructor.name, message: err.message }
  }
}

function main () {
  const rng = new Rng(0x2545f4914f6cdd1dn)
  const inputs = []

  // numeric inputs: the expression is JSON of the double, so the Python test
  // gets the exact same bits with `float(json.loads(expr))`
  for (let i = 0; i < 1200; i++) {
    const v = rng.longDecimal()
    inputs.push({ expr: JSON.stringify(v) })
  }
  for (let i = 0; i < 1200; i++) {
    const v = rng.shortDecimal()
    inputs.push({ expr: JSON.stringify(v) })
  }
  // exact rationals the README calls out, and negative zero
  const exact = ['-0', '0.1 + 0.2', '1 / 3', '2 / 3', '7 / 9', '512 / 999',
    '2.555', '-2.555', '8.36', '-8.36', '2 ** 53', '2 ** 53 + 2',
    '1e21', '-1e21', '1e-7', '1e-6', '1e300', '1e-310', '5e-324',
    '1.7976931348623157e308', '41033158946.448006', '70142542814845430',
    // the doubles around 0.5 where `floor(x + 0.5)` and Math.round differ
    '0.5 - 2 ** -53', '-0.5 + 2 ** -53', '0.5 - 2 ** -54', '-0.5 + 2 ** -54',
    '0.5', '-0.5', '2.5', '-2.5', '-0.4', '-1.4']
  for (const expr of exact) inputs.push({ expr: `(${expr})` })
  // strings: upstream runs them through parseFloat
  for (const s of ['0', '-0', '1.5px', '3em', '3e3', '1/2', '.5', '5.', '+1.5',
    ' 7 ', '0x10', '.2', '1.1', '-1.1', '8.36', 'abc', '', '   ']) {
    inputs.push({ expr: JSON.stringify(s) })
  }
  // negatives and ties across magnitudes: the region where Math.round's
  // treatment of -0 and of halves is visible through the whole pipeline
  for (let k = 0; k < 400; k++) {
    const base = (k % 200) - 100
    for (const frac of [0.5, 0.49999999999999994, 0.5000000000000001, 0.25,
      0.75, 0.1]) {
      inputs.push({ expr: `(${base} + ${frac})` })
    }
  }
  for (let e = -6; e <= 6; e++) {
    for (const f of [0.4, 0.5, 2.5, -0.4, -0.5, -2.5, 4.5, -4.5, -0.25]) {
      inputs.push({ expr: `(${f} * 10 ** ${e})` })
    }
  }
  // non-number values upstream coerces with Number(x)
  for (const expr of ['[0.5]', '[0.5, 0.25]', 'true', 'false', 'null', '{}']) {
    inputs.push({ expr })
  }

  for (const item of inputs) {
    const value = eval(item.expr) // eslint-disable-line no-eval
    // upstream: `if (typeof num === 'string') num = parseFloat(num)`, and
    // findPrecision coerces every other non-number with Number(num).
    let parsed
    if (typeof value === 'string') parsed = parseFloat(value)
    else if (typeof value === 'number') parsed = value
    else parsed = Number(value)
    const rec = probe(item.expr, item.expr, parsed)
    if (typeof value === 'string') rec.raw = value
    process.stdout.write(JSON.stringify(rec) + '\n')
  }
}

main()
