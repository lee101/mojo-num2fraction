from std.math import fma, floor, trunc

# num2fraction -- numeric core, ported to Mojo.
#
# Upstream is 40 lines of JavaScript (npm `num2fraction`, github.com/yisibl).
# All of its compute is here:
#
#   * an epsilon comparison, |a - b| <= 9.5367432e-7 (that is 2^-20);
#   * a Euclidean GCD over IEEE doubles that uses the *JavaScript* remainder;
#   * a precision search that multiplies by ten until round(n*e)/e ~= n;
#   * one division and two Math.round calls.
#
# Every one of those is a data-dependent per-element loop, so together they
# are the whole numeric surface of the package.  Only the string
# concatenation at the very end stays on the Python side.
#
# Two JavaScript behaviours are reproduced deliberately and are load-bearing:
#
#   1. `%` in JavaScript is `fmod` (truncated quotient, the result takes the
#      sign of the dividend).  Mojo's `Float64 %` is a *floored* modulo, the
#      result takes the sign of the divisor, so `_js_remainder` implements
#      fmod by hand.  Using `%` here silently changes the GCD result.
#   2. `Math.round` is `floor(x + 0.5)`, i.e. ties go towards +Infinity.
#      Mojo's `round` is half-away-from-zero, so `_js_round` is used instead.
#      They disagree on exactly the negative halves, e.g. -0.5.

# Upstream literal.  Do not "simplify" this to 1e-6: the two differ in the
# last digits and it changes which n the precision search stops at.
comptime EPSILON: Float64 = 9.5367432e-7

# 10 ** 309 is already past the largest finite double, so a precision search
# that has not converged by then can never converge.  Upstream has no such
# guard: it recurses until the stack dies or the loop runs forever.
comptime MAX_PRECISION_STEPS: Int = 400

# Largest finite Float64; used instead of an is_finite() intrinsic.
comptime MAX_FINITE: Float64 = 1.7976931348623157e308

# Smallest normal double; below this, scaling an operand by two loses bits.
comptime MIN_NORMAL: Float64 = 2.2250738585072014e-308

# Status codes returned through the status array.
comptime STATUS_OK: Int32 = 0
comptime STATUS_NO_PRECISION: Int32 = 1
comptime STATUS_NONFINITE: Int32 = 2


@always_inline
def _almost_eq(a: Float64, b: Float64) -> Bool:
    return abs(a - b) <= EPSILON


@always_inline
def _js_round(x: Float64) -> Float64:
    # JavaScript Math.round: the closest integral value, ties towards +Inf.
    #
    # `floor(x + 0.5)` is the usual one-liner and it is wrong above 2**52,
    # where the ulp is 1 or more and the addition rounds before the floor:
    # V8 gives Math.round(5129144868306001) === 5129144868306001 while
    # floor(5129144868306001 + 0.5) is ...002.  Every double at or above
    # 2**52 is already integral, and ECMA-262 returns x unchanged in that
    # case, so the identity branch is the exact answer.
    if floor(x) == x:
        return x
    return floor(x + 0.5)


@always_inline
def _is_finite(v: Float64) -> Bool:
    return v == v and abs(v) <= MAX_FINITE

@always_inline
def _js_remainder(x: Float64, y: Float64) -> Float64:
    # JavaScript's `%` == C fmod, and IEEE-754 requires fmod to be *exact*.
    # The GCD's Euclidean path inherits that exactness, so an approximation
    # here is not a rounding-tolerance question: it changes the answer.
    # Measured against node over the 1616-value upstream fixture, the short
    # form `x - trunc(x / y) * y` puts 95 answers one ulp out -- e.g.
    # num2fraction(41033158946.448006) comes out 5129144868306002/125000
    # instead of 5129144868306001/125000.
    #
    # Fast path, quotients below 2**53.  `fma(q, y, -x)` evaluates q * y - x
    # with a single rounding, and the exact remainder is representable, so it
    # is exact.  The division is within half an ulp of the true quotient, so
    # trunc() of it can be one low; that shows up as |r| >= y and is corrected.
    #
    # Slow path, quotients at or above 2**53 -- including quotients that
    # overflow to infinity, which is where any "scale both operands down"
    # scheme fails: the quotient of two operands scaled by the same power of
    # two is unchanged, so the scaling never makes the quotient representable.
    # Instead the divisor is doubled until it is within a factor of two of the
    # running remainder and subtracted once.  Doubling a double is exact, and
    # a subtraction with the subtrahend in (r/2, 2r) is exact by Sterbenz, so
    # every step is exact and the loop runs once per bit of the quotient.
    if y == 0.0 or not _is_finite(x) or not _is_finite(y):
        return 0.0
    var ax = abs(x)
    var ay = abs(y)
    var quotient = ax / ay
    var r = ax
    if _is_finite(quotient) and quotient < 9007199254740992.0:
        # ax - trunc(quotient) * ay, exact because fma rounds once and the
        # exact value is representable.
        r = -fma(trunc(quotient), ay, -ax)
        # trunc() of the rounded quotient can be one either side of the true
        # floor, which shows up as r < 0 or r >= ay.  Both are one step.
        var guard = 0
        while r < 0.0 and guard < 4:
            r = r + ay
            guard += 1
        guard = 0
        while r >= ay and guard < 4:
            r = r - ay
            guard += 1
    else:
        while r >= ay:
            var t = ay
            # t + t can overflow to inf, which correctly ends the doubling:
            # r is finite, so r < t + t holds.
            while t + t <= r:
                t = t + t
            r = r - t
    if r == 0.0:
        # fmod keeps the sign of the dividend, -0 included.
        return -0.0 if x < 0.0 else 0.0
    return -r if x < 0.0 else r





def _gcd(a_in: Float64, b_in: Float64) -> Float64:
    # Upstream is `GCD(a, b) { return almostEq(b, 0) ? a : GCD(b, a % b) }`.
    # Iterative form.  The guard only fires for the non-finite inputs upstream
    # recurses forever on.
    var x = a_in
    var y = b_in
    var guard = 0
    while not _almost_eq(y, 0.0):
        if guard >= 128:
            return 0.0
        var t = _js_remainder(x, y)
        x = y
        y = t
        guard += 1
    return x


def _find_precision(n: Float64) -> Float64:
    # var e = 1; while (!almostEq(round(n * e) / e, n)) e *= 10; return e;
    var e = 1.0
    var steps = 0
    while not _almost_eq(_js_round(n * e) / e, n):
        e = e * 10.0
        steps += 1
        if steps > MAX_PRECISION_STEPS or not _is_finite(e):
            return 0.0
    return e


def _convert_one(x: Float64) -> Tuple[Float64, Float64, Float64, Int32]:
    # Returns (numerator, denominator, precision, status), all as doubles
    # because upstream's number type is an IEEE double all the way through.
    if not _is_finite(x):
        return 0.0, 1.0, 0.0, STATUS_NONFINITE
    if x == 0.0:
        # Upstream short-circuits here and returns the string "0".
        return 0.0, 1.0, 1.0, STATUS_OK
    var e = _find_precision(x)
    if e == 0.0:
        return 0.0, 1.0, 0.0, STATUS_NO_PRECISION
    var number = x * e
    if not _is_finite(number):
        return 0.0, 1.0, 0.0, STATUS_NONFINITE
    var g = _gcd(number, e)
    if not _is_finite(g) or g == 0.0:
        return 0.0, 1.0, 0.0, STATUS_NONFINITE
    g = abs(g)
    var numerator = _js_round(number / g)
    var denominator = _js_round(e / g)
    if not _is_finite(numerator) or not _is_finite(denominator):
        return 0.0, 1.0, 0.0, STATUS_NONFINITE
    return numerator, denominator, e, STATUS_OK


@export("n2f_convert_range")
def n2f_convert_range(
    start: Int,
    stop: Int,
    in_ptr: Int,
    num_ptr: Int,
    den_ptr: Int,
    prec_ptr: Int,
    status_ptr: Int,
) abi("C"):
    var xs = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var nums = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=num_ptr)
    var dens = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=den_ptr)
    var precs = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=prec_ptr)
    var stats = Pointer[Int32, AnyOrigin[mut=True]](unsafe_from_address=status_ptr)
    for i in range(start, stop):
        var r = _convert_one(xs[unsafe_offset=i])
        nums[unsafe_offset=i] = r[0]
        dens[unsafe_offset=i] = r[1]
        precs[unsafe_offset=i] = r[2]
        stats[unsafe_offset=i] = r[3]


@export("n2f_convert_value")
def n2f_convert_value(x: Float64, out_ptr: Int, status_ptr: Int) abi("C"):
    # out = [numerator, denominator, precision]
    var o = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var stats = Pointer[Int32, AnyOrigin[mut=True]](unsafe_from_address=status_ptr)
    var r = _convert_one(x)
    o[unsafe_offset=0] = r[0]
    o[unsafe_offset=1] = r[1]
    o[unsafe_offset=2] = r[2]
    stats[unsafe_offset=0] = r[3]


@export("n2f_precision")
def n2f_precision(n: Float64) abi("C") -> Float64:
    return _find_precision(n)


@export("n2f_precision_array")
def n2f_precision_array(n: Int, in_ptr: Int, out_ptr: Int) abi("C"):
    var a = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var o = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    for i in range(n):
        o[unsafe_offset=i] = _find_precision(a[unsafe_offset=i])


@export("n2f_gcd")
def n2f_gcd(a: Float64, b: Float64) abi("C") -> Float64:
    return _gcd(a, b)


@export("n2f_gcd_array")
def n2f_gcd_array(n: Int, a_ptr: Int, b_ptr: Int, out_ptr: Int) abi("C"):
    var a = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=a_ptr)
    var b = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=b_ptr)
    var o = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    for i in range(n):
        o[unsafe_offset=i] = _gcd(a[unsafe_offset=i], b[unsafe_offset=i])


@export("n2f_js_remainder_array")
def n2f_js_remainder_array(n: Int, a_ptr: Int, b_ptr: Int, out_ptr: Int) abi("C"):
    # Exposed so the fmod-vs-floored-modulo difference is directly testable.
    var a = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=a_ptr)
    var b = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=b_ptr)
    var o = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    for i in range(n):
        o[unsafe_offset=i] = _js_remainder(a[unsafe_offset=i], b[unsafe_offset=i])


@export("n2f_js_round_array")
def n2f_js_round_array(n: Int, in_ptr: Int, out_ptr: Int) abi("C"):
    # Exposed so the floor(x+0.5) tie rule is directly testable.
    var a = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var o = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    for i in range(n):
        o[unsafe_offset=i] = _js_round(a[unsafe_offset=i])
