"""Validated lognormal density and positive-weight Taylor quadrature."""
from fractions import Fraction as F
from functools import lru_cache
from .interval import I,exp_point,log,sqrt,pi
from .polynomial import bump,mul,integrate
from .progress import QUIET


def square(x):
    vals=(x.lo*x.lo,x.hi*x.hi)
    return I(0 if x.lo<=0<=x.hi else min(vals),max(vals))


def negexp(x):
    # A tiny positive upper enclosure avoids underflow for distant wings.
    if x.hi < -90:return I(0,exp_point(F(-90)).hi)
    return I(0 if x.lo < -90 else exp_point(x.lo).lo,exp_point(x.hi).hi)


@lru_cache(maxsize=32768)
def gamma_box(ta,tb,ka,kb,variance):
    """Return G=k² c_kk and its first/second derivatives, all enclosed."""
    if ta<=0 or ka<=0 or variance<=0:raise ValueError('Positive density coordinates required')
    v=I(variance*ta,variance*tb);k=I(ka,kb);ell=log(k)
    g=(sqrt(k)/sqrt(2*pi()*v)*negexp(-square(ell)/(2*v)-v/8)).rounded()
    lv=(-1/(2*v)+square(ell)/(2*v*v)-F(1,8)).rounded()
    lvv=(1/(2*v*v)-square(ell)/(v*v*v)).rounded()
    lk=((F(1,2)-ell/v)/k).rounded()
    lkk=((-F(1,2)+(ell-1)/v)/(k*k)).rounded()
    lvk=(ell/(v*v*k)).rounded()
    return (g,(variance*g*lv).rounded(),(g*lk).rounded(),
            (variance*variance*g*(square(lv)+lvv)).rounded(),
            (variance*g*(lv*lk+lvk)).rounded(),(g*(square(lk)+lkk)).rounded())


def absolute(x):return max(abs(x.lo),abs(x.hi))


def moments(p,a,b):
    m=(a+b)/2
    first=mul(p,[-m,1]);second=mul(p,[m*m,-2*m,1])
    return (integrate(p,a,b),integrate(first,a,b),integrate(second,a,b),
            -integrate(first,a,m)+integrate(first,m,b))


def density_integral(window,variance,cells=16,progress=QUIET):
    """Enclose ∫phi k²c_kk using midpoint value/gradient and Hessian remainder.

    The polynomial kernel moments are exact. No difference-of-grids estimate
    or floating quadrature tolerance is used as a certificate.
    """
    w=window;v=F(variance)
    pt,pk=bump(w.t,w.ht),bump(w.k,w.hk)
    ts=[w.t-w.ht+2*w.ht*i/cells for i in range(cells+1)]
    ks=[w.k-w.hk+2*w.hk*j/cells for j in range(cells+1)]
    tm=[moments(pt,a,b) for a,b in zip(ts[:-1],ts[1:])]
    km=[moments(pk,a,b) for a,b in zip(ks[:-1],ks[1:])]
    result=I(0)
    for i,(ta,tb) in enumerate(zip(ts[:-1],ts[1:])):
        for j,(ka,kb) in enumerate(zip(ks[:-1],ks[1:])):
            t0,k0=(ta+tb)/2,(ka+kb)/2
            g,gt,gk,*_=gamma_box(t0,t0,k0,k0,v)
            gb,_,_,htt,htk,hkk=gamma_box(ta,tb,ka,kb,v)
            a0,a1,a2,aa=tm[i];b0,b1,b2,bb=km[j]
            central=g*(a0*b0)+gt*(a1*b0)+gk*(a0*b1)
            error=absolute(htt)*a2*b0/2+absolute(htk)*aa*bb+absolute(hkk)*a0*b2/2
            box=(central+I(-error,error)).intersect(I(gb.lo*a0*b0,gb.hi*a0*b0))
            result=(result+box).rounded()
        progress.event('witness_density_quadrature','running',done=(i+1)*cells,total=cells*cells)
    if result.lo<=0:raise ArithmeticError('Density integral lower bound not positive; refine quadrature')
    return result
