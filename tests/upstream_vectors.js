// Regenerates tests/upstream_vectors.json by running the REAL upstream module.
//
//   /home/administrator/.local/bin/uv pip download --no-deps -d /tmp/up num2fraction  # (npm, see README)
//   node tests/upstream_vectors.js /tmp/up/n2f_index.js > tests/upstream_vectors.json
//
// Upstream's findPrecision has no iteration cap, so a value that cannot
// converge spins forever in node.  The generator wraps the reference call in
// a convergence pre-check and records those values separately instead of
// hanging; the Mojo port reports them as Num2FractionError.

'use strict'
const fs = require('fs')
const n2f = require(process.argv[2])

var EPSILON = 9.5367432e-7

function almostEq (a, b) { return Math.abs(a - b) <= EPSILON }

function converges (n) {
  var e = 1
  for (var i = 0; i < 400; i++) {
    if (almostEq(Math.round(n * e) / e, n)) return true
    e = e * 10
    if (!isFinite(e)) return false
  }
  return false
}

// deterministic 64-bit LCG (BigInt: a double LCG would lose the low bits and
// collapse the generated set to a handful of values)
var state = 0x2545f4914f6cdd1dn
var M64 = (1n << 64n) - 1n
function rnd () {
  state = (state * 6364136223846793005n + 1442695040888963407n) & M64
  return Number(state >> 11n) / 9007199254740992
}

var cases = []
function add (x) { cases.push(x) }


;[
  0, 0.2, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 2.1, 3, 2.555, 8.36, 7 / 9, 8 / 9,
  512 / 999, (2 * Math.PI / 3) / Math.PI, (8 * 5) / (4 / 2), 0.5, 0.1, 0.3,
  0.7, 1 / 3, 2 / 3, 1.005, 2.675, 99.95, 1e-7, 1e-3, 1e21, 1e-21,
  1.7976931348623157e308, 1e300
].forEach(add)

// negative ties: Math.round is floor(x+0.5), so these must keep the sign
;[-0.5, -1.5, -0.25, -0.125, -2.5, -0.75, -2.675, -99.95].forEach(add)

// Known non-terminating inputs, verified with `timeout 3 node`: Infinity, NaN
// and -Infinity each exit 124 (killed), because Math.round(inf * e) / e is
// NaN and therefore never within the epsilon of n.  No *finite* input was
// found that fails to converge: a 300k-value sweep of non-integral doubles
// between 1e4 and 1e18 always found an e whose product rounds well enough.
;[Infinity, NaN, -Infinity].forEach(add)

// the epsilon boundary: round(n*e)/e must land within 2^-20 of n
;[1e-6, 9.5367432e-7, 9.5367432e-7 * 0.99, 1e-6 * 1.01, 5e-7, 2e-6].forEach(add)

// exact binary fractions and powers of ten
;[
  0.5, 0.25, 0.75, 0.0625, 1 / 1024, 3 / 4, 5 / 8, 0.3333333333333333,
  2.718281828459045, 3.14159265358979, 1234.5678, 0.123456789,
  0.1234567890123456789, 123456789.123456789, 2 ** 40, 2 ** 40 + 0.5,
  1e-300, 1e300, 1e-310
].forEach(add)

// a spread of magnitudes and digit counts
for (var i = 0; i < 1500; i++) {
  var digits = 1 + (i % 17)
  var mag = Math.floor(rnd() * 17) - 6
  var s = (rnd() * 9 + 1).toFixed(digits)
  var x = parseFloat(s) * Math.pow(10, mag)
  if (rnd() < 0.5) x = -x
  add(x)
}

// integers, including ones past the point where JS switches to exponent form
for (var k = 0; k < 60; k++) add(Math.floor(rnd() * Math.pow(10, 1 + (k % 22))))

var out = []
var skipped = []
var seen = {}
cases.forEach(function (x) {
  var key = Object.is(x, -0) ? '-0' : String(x)
  if (seen[key]) return
  seen[key] = 1
  if (!isFinite(x) || !converges(x)) { skipped.push(key); return }
  out.push({ in: key, out: n2f(x) })
})

process.stdout.write(JSON.stringify({
  source: 'npm num2fraction 1.2.2 (github.com/yisibl/num2fraction), run under node',
  cases: out,
  nonterminating: skipped
}, null, 1) + '\n')
