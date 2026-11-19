"""Exact polynomial pieces for test functions and compact cubic splines."""
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
from math import comb
from .polynomial import add,mul,deriv,integrate,bump,compose_affine,bernstein_range


def value(p,x):
    y=F(0)
    for c in reversed(p):y=y*x+c
    return y


def bernstein(p,lo,hi,degree=None):
    q=compose_affine(p,lo,hi-lo)
    n=max(len(q)-1,degree or 0)
    return [sum(q[j]*F(comb(i,j),comb(n,j)) for j in range(min(i,len(q)-1)+1)) for i in range(n+1)]


@lru_cache(maxsize=16384)
def spline_piece(knots,index,degree,cell):
    if cell<index or cell>index+degree:return (F(0),)
    if degree==0:return (F(1),) if cell==index else (F(0),)
    a=knots[index+degree]-knots[index]
    b=knots[index+degree+1]-knots[index+1]
    left=mul([-knots[index]/a,1/a],spline_piece(knots,index,degree-1,cell))
    right=mul([knots[index+degree+1]/b,-1/b],spline_piece(knots,index+1,degree-1,cell))
    return tuple(add(left,right))


@dataclass(frozen=True)
class TestCurve:
    left:F
    right:F
    middle:tuple
    above:tuple
    kind:str

    def piece(self,l,h=None,order=0):
        p=self.above if l>=self.right else ((F(0),) if h is not None and h<=self.left else self.middle)
        p=list(p)
        for _ in range(order):p=deriv(p)
        return p

    def at(self,x,order=0):
        if x<self.left:return F(0)
        p=self.above if x>=self.right else self.middle
        for _ in range(order):p=deriv(p)
        return value(p,x)


def test_curve(left,right,kind='call'):
    center=(left+right)/2;half=(right-left)/2
    p=bump(center,half)
    mass=integrate(p,left,right)
    p=[x/mass for x in p]
    if kind=='bump':return TestCurve(left,right,tuple(p),(F(0),),'bump')
    digital=[F(0)]+[c/(i+1) for i,c in enumerate(p)]
    digital[0]=-value(digital,left)
    if kind=='digital':return TestCurve(left,right,tuple(digital),(F(1),),'digital')
    call=[F(0)]+[c/(i+1) for i,c in enumerate(digital)]
    call[0]=-value(call,left)
    return TestCurve(left,right,tuple(call),(-center,F(1)),'call')


def moment_enclosure(curve,left,right):
    """f(x)=a+b*x+c*x²+remainder on a cell, with exact remainder range."""
    p=curve.piece(left,right)
    if right is None or len(p)<=3:
        return tuple(list(p)+[F(0)]*(3-len(p))),F(0),F(0)
    points=(left,(left+right)/2,right)
    interp=[F(0)]
    for i,x in enumerate(points):
        term=[F(1)];den=F(1)
        for j,y in enumerate(points):
            if i!=j:term=mul(term,[-y,F(1)]);den*=x-y
        interp=add(interp,[z*value(p,x)/den for z in term])
    residual=add(p,[-x for x in interp])
    r=bernstein_range(residual,left,right)
    return tuple(interp+[F(0)]*(3-len(interp))),r.lo,r.hi
