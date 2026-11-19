"""Small rational interval engine; trusted path uses no libm/GPU rounding.

Endpoints and all arithmetic are fractions. Transcendental bounds use explicit
series remainders. This is intentionally slower than the numerical path.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
from math import isqrt, factorial
from .display import approximate

BITS = 110
TINY = F(1, 2**BITS)


def frac(x):
    if isinstance(x, F): return x
    if isinstance(x, float): return F.from_float(x)
    return F(x)


@dataclass(frozen=True)
class I:
    lo: F
    hi: F

    def __init__(self, lo, hi=None):
        object.__setattr__(self, "lo", frac(lo))
        object.__setattr__(self, "hi", frac(lo if hi is None else hi))
        if self.lo > self.hi:
            raise ValueError("Empty interval")

    def __add__(self, other):
        other = as_i(other)
        return I(self.lo+other.lo, self.hi+other.hi)
    __radd__ = __add__

    def __neg__(self): return I(-self.hi, -self.lo)
    def __sub__(self, other): return self+-as_i(other)
    def __rsub__(self, other): return as_i(other)+-self

    def __mul__(self, other):
        o = as_i(other)
        p = (self.lo*o.lo, self.lo*o.hi, self.hi*o.lo, self.hi*o.hi)
        return I(min(p), max(p))
    __rmul__ = __mul__

    def __truediv__(self, other):
        o = as_i(other)
        if o.lo <= 0 <= o.hi: raise ZeroDivisionError("Interval contains zero")
        return self*I(1/o.hi, 1/o.lo)

    def __rtruediv__(self, other): return as_i(other)/self

    def __pow__(self, n):
        if n < 0: return 1/(self**(-n))
        result = I(1)
        for _ in range(n): result = result*self
        return result

    def hull(self, o):
        o = as_i(o)
        return I(min(self.lo, o.lo), max(self.hi, o.hi))

    def intersect(self, o):
        o = as_i(o)
        return I(max(self.lo, o.lo), min(self.hi, o.hi))

    def rounded(self, bits=BITS):
        """Outward dyadic compression prevents growth of rational expressions."""
        scale = 2**bits
        lo = self.lo.numerator*scale//self.lo.denominator
        hi = -((-self.hi.numerator*scale)//self.hi.denominator)
        return I(F(lo, scale), F(hi, scale))

    def json(self):
        return dict(lo=str(self.lo), hi=str(self.hi),
                    approximate=[approximate(self.lo), approximate(self.hi)])


def as_i(x): return x if isinstance(x, I) else I(x)


@lru_cache(maxsize=8192)
def exp_point(x):
    x = frac(x)
    if x < 0: return (1/exp_point(-x)).rounded()
    if x > 100: raise ValueError("Rational exp input exceeds 100; rescale the problem")
    term = F(1)
    total = term
    n = 0
    while True:
        n += 1
        term = term*x/n
        total += term
        next_term = term*x/(n+1)
        ratio = x/(n+2)
        if ratio < 1:
            remainder = next_term/(1-ratio)
            if remainder < TINY:
                return I(total, total+remainder).rounded()


def exp(x):
    x = as_i(x)
    return I(exp_point(x.lo).lo, exp_point(x.hi).hi)


@lru_cache(maxsize=8192)
def sqrt_point(x):
    x = frac(x)
    if x < 0: raise ValueError("Negative square root")
    scale = 2**BITS
    a = isqrt((x.numerator*scale*scale)//x.denominator)
    lo = F(a, scale)
    return I(lo, lo if lo*lo == x else F(a+1, scale))


def sqrt(x):
    x = as_i(x)
    return I(sqrt_point(x.lo).lo, sqrt_point(x.hi).hi)


def _log_unit(x):
    z = (x-1)/(x+1)
    negative = z < 0
    z = abs(z)
    z2 = z*z
    power = z
    total = F(0)
    n = 0
    while True:
        total += power/(2*n+1)
        power *= z2
        n += 1
        remainder = 2*power/((2*n+1)*(1-z2))
        if remainder < TINY:
            result = I(2*total, 2*total+remainder)
            return (-result if negative else result).rounded()


@lru_cache(maxsize=8192)
def log_point(x):
    x = frac(x)
    if x <= 0: raise ValueError("Nonpositive logarithm")
    power = 0
    while x > 2: x /= 2; power += 1
    while x < 1: x *= 2; power -= 1
    return (_log_unit(x)+power*_log_unit(F(2))).rounded()


def log(x):
    x = as_i(x)
    return I(log_point(x.lo).lo, log_point(x.hi).hi)


@lru_cache(maxsize=1)
def pi():
    def atan_inverse(n):
        z = F(1, n)
        term = z
        result = F(0)
        j = 0
        while True:
            result += (-1)**j*term/(2*j+1)
            j += 1
            term *= z*z
            rem = term/(2*j+1)
            if rem < TINY/32:
                return I(result-rem, result+rem)
    return (16*atan_inverse(5)-4*atan_inverse(239)).rounded()


def _cdf_point(x):
    x = frac(x)
    if x < 0: return (1-_cdf_point(-x)).rounded()
    inv_sqrt = (1/sqrt(2*pi())).rounded()
    if x >= 12:
        # Mills' bound decreases on x>0. Capping its argument at 14 keeps
        # exp within a useful finite range even for extremely deep wings.
        z=min(x,F(14))
        tail = (exp(-z*z/2)*inv_sqrt/z).hi
        return I(1-tail, 1).rounded()
    # Integrate the Taylor series for exp(-u^2/2). The integral of the
    # Lagrange remainder is <= x^(2n+3)/(2^(n+1)(n+1)!(2n+3)).
    n = 0
    power = x
    total = x
    while True:
        n += 1
        power *= x*x
        term = power/(2**n*factorial(n)*(2*n+1))
        total += (-1)**n*term
        rem = power*x*x/(2**(n+1)*factorial(n+1)*(2*n+3))
        if rem < TINY:
            integ = I(total-rem, total+rem)
            return (F(1, 2)+integ*inv_sqrt).intersect(I(0, 1)).rounded()


def normal_cdf(x):
    x = as_i(x)
    return I(_cdf_point(x.lo).lo, _cdf_point(x.hi).hi)


@lru_cache(maxsize=8192)
def black_call(k, integrated_variance):
    k, v = frac(k), frac(integrated_variance)
    if k <= 0 or v < 0: raise ValueError("Invalid Black input")
    if v == 0: return I(max(0, 1-k))
    root = sqrt(v)
    d1 = ((-log(k)+v/2)/root).rounded()
    d2 = (d1-root).rounded()
    return (normal_cdf(d1)-k*normal_cdf(d2)).rounded().intersect(I(max(0, 1-k), 1))


def sin_point(x):
    """Rational sine bounds for |x| <= 8 (remainder <= |x|^(2n+3)/(2n+3)!)."""
    x = frac(x)
    if abs(x) > 8: raise ValueError("sin: reduce the argument to [-8,8]")
    n = 0
    total = x
    term = x
    while True:
        n += 1
        term *= -x*x/((2*n)*(2*n+1))
        total += term
        rem = abs(x)**(2*n+3)/factorial(2*n+3)
        if rem < TINY: return I(total-rem, total+rem).rounded()


def sin_interval_lipschitz(x):
    x = as_i(x)
    mid = (x.lo+x.hi)/2
    radius = (x.hi-x.lo)/2
    return (sin_point(mid)+I(-radius, radius)).intersect(I(-1, 1)).rounded()
