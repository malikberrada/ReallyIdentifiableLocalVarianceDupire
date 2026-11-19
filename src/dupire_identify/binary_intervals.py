"""Outward IEEE-754 binary64 intervals, with one ufunc per operation.

Stored binary64 candidate coefficients denote exact rational numbers. These
intervals enclose computations on those rationals; no libm approximation,
BLAS contraction, finite-difference error estimate or optimizer flag is used.
The rational interval engine remains the reference for transcendental data.
"""
from fractions import Fraction as F
import math
import numpy as np


def down(x): return np.nextafter(x, -np.inf)
def up(x): return np.nextafter(x, np.inf)


class B:
    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(lo if hi is None else hi, dtype=np.float64)
        if np.any(~np.isfinite(self.lo)) or np.any(~np.isfinite(self.hi)) or np.any(self.lo > self.hi):
            raise ValueError('Nonfinite or reversed binary interval')

    @classmethod
    def rational(cls, value):
        q = F(value); f = float(q); represented = F.from_float(f)
        return cls(np.nextafter(f, -np.inf) if represented > q else f,
                   np.nextafter(f, np.inf) if represented < q else f)

    def __add__(self, other):
        o = as_b(other)
        return B(down(self.lo + o.lo), up(self.hi + o.hi))
    __radd__ = __add__

    def __neg__(self): return B(-self.hi, -self.lo)
    def __sub__(self, other): return self + -as_b(other)
    def __rsub__(self, other): return as_b(other) + -self

    def __mul__(self, other):
        o = as_b(other)
        products = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi]
        return B(down(np.minimum.reduce(products)), up(np.maximum.reduce(products)))
    __rmul__ = __mul__

    def __truediv__(self, other):
        o = as_b(other)
        if np.any((o.lo <= 0) & (o.hi >= 0)): raise ZeroDivisionError('Interval contains zero')
        return self * B(down(1. / o.hi), up(1. / o.lo))
    def __rtruediv__(self, other): return as_b(other) / self

    def __getitem__(self, key): return B(self.lo[key], self.hi[key])

    def absolute_upper(self): return np.maximum(np.abs(self.lo), np.abs(self.hi))


def as_b(x): return x if isinstance(x, B) else B(x)


def stack(xs, axis=0):
    return B(np.stack([as_b(x).lo for x in xs], axis=axis),
             np.stack([as_b(x).hi for x in xs], axis=axis))


def elevate(coefficients, degree):
    """Univariate Bernstein degree elevation, along the last axis."""
    c = as_b(coefficients)
    n = c.lo.shape[-1] - 1
    while n < degree:
        rows = [c[..., 0]]
        for j in range(1, n + 1):
            w = B.rational(F(j, n + 1))
            rows.append(w*c[..., j-1] + (1-w)*c[..., j])
        rows.append(c[..., n]); c = stack(rows, axis=-1); n += 1
    if n != degree: raise ValueError('Cannot reduce Bernstein degree')
    return c


def product(a, b):
    """Product of Bernstein polynomials along the last axis."""
    a, b = as_b(a), as_b(b)
    n, m = a.lo.shape[-1]-1, b.lo.shape[-1]-1
    out = []
    for k in range(n+m+1):
        terms = []
        for i in range(max(0,k-m), min(n,k)+1):
            w = B.rational(F(math.comb(n,i)*math.comb(m,k-i), math.comb(n+m,k)))
            terms.append(w*a[...,i]*b[...,k-i])
        z = terms[0]
        for term in terms[1:]: z = z+term
        out.append(z)
    return stack(out, axis=-1)


def split(c, parameter):
    c=as_b(c); u=as_b(parameter); levels=[c]; left=[c[...,0]]; right=[c[...,-1]]
    for unused in range(c.lo.shape[-1]-1):
        c=(1-u[...,None])*c[...,:-1]+u[...,None]*c[...,1:]
        left.append(c[...,0]);right.append(c[...,-1])
    return stack(left,axis=-1),stack(right[::-1],axis=-1)


def restrict(c, left, right):
    """Restrict Bernstein coefficients from [0,1] to [left,right]."""
    l,r=as_b(left),as_b(right)
    prefix,_=split(c,r)
    _,answer=split(prefix,l/r)
    return answer


def self_check():
    if np.finfo(np.float64).nmant!=52 or np.finfo(np.float64).maxexp!=1024:
        raise RuntimeError('Binary64 IEEE arithmetic required')
    if np.nextafter(1.,np.inf)!=1.+2.**-52 or np.add(np.array([1.]),np.array([2.**-53]))[0]!=1.:
        raise RuntimeError('Round-to-nearest binary64 required')
    tiny=np.nextafter(0.,1.)
    if np.add(np.array([tiny]),np.array([tiny]))[0]!=np.nextafter(tiny,1.):
        raise RuntimeError('Gradual subnormal arithmetic required')
    return True


def sum_all(value):
    """Pairwise outward sums, avoiding unbounded BLAS/reduction roundoff."""
    value=as_b(value);lo=value.lo.ravel();hi=value.hi.ravel()
    if not len(lo):return B(0.)
    while len(lo)>1:
        n=len(lo)//2
        left=down(lo[:2*n:2]+lo[1:2*n:2]);right=up(hi[:2*n:2]+hi[1:2*n:2])
        if len(lo)%2:left=np.r_[left,lo[-1]];right=np.r_[right,hi[-1]]
        lo,hi=left,right
    return B(lo[0],hi[0])
