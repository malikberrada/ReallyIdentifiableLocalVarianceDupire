"""P1 outer bounds covering ALL bounded-variance continuous martingales.

This is an enclosure, not an assertion of sharpness or attainability. There is
no fitted spline, frozen boundary condition or finite-difference PDE in this
trusted path. Compact support removes spatial-tail terms in the weak identity.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from .interval import I, exp_point, sqrt_point
from .polynomial import bump, deriv, mul, integrate, bernstein_range
from .io import quote_payload, rational
from .progress import QUIET


@dataclass(frozen=True)
class Window:
    t: F
    k: F
    ht: F
    hk: F

    @classmethod
    def make(cls, t, k, ht, hk):
        w = cls(*(rational(v) for v in (t,k,ht,hk)))
        if min(w.ht,w.hk) <= 0 or w.t-w.ht <= 0 or w.k-w.hk <= 0:
            raise ValueError("Kernel support must lie in t>0, k>0")
        return w

    def json(self): return {key:str(getattr(self,key)) for key in ("t","k","ht","hk")}


def price_envelope(quotes, ta, tb, ka, kb, amax):
    """Bounds valid at every point of a rectangle, from monotonicity and L2.

    E[(X_v-X_u)^2] <= exp(amax*v)-exp(amax*u), u<=v.
    A strike increment lowers a call by at most that increment.
    """
    lo = max(F(0), 1-kb)
    hi = F(1)
    for q in quotes:
        lag = F(0)
        lead = F(0)
        if ta < q.t:
            v = max(F(0), exp_point(amax*q.t).hi-exp_point(amax*ta).lo)
            lag = sqrt_point(v).hi
        if tb > q.t:
            v = max(F(0), exp_point(amax*tb).hi-exp_point(amax*q.t).lo)
            lead = sqrt_point(v).hi
        lo = max(lo, q.bid-max(F(0),kb-q.k)-lag)
        hi = min(hi, q.ask+max(F(0),q.k-ka)+lead)
    return I(lo,hi)


def weighted_cell(p, q, ta, tb, ka, kb, c):
    """Integral enclosure of p(t) q(k) c(t,k), with exact signed weight."""
    mass = integrate(p,ta,tb)*integrate(q,ka,kb)
    weight = bernstein_range(p,ta,tb)*bernstein_range(q,ka,kb)
    if weight.lo >= 0: return I(mass*c.lo,mass*c.hi)
    if weight.hi <= 0: return I(mass*c.hi,mass*c.lo)
    volume = (tb-ta)*(kb-ka)
    delta = c.hi-c.lo
    return I(c.lo*mass+delta*min(F(0),weight.lo)*volume,
             c.lo*mass+delta*max(F(0),weight.hi)*volume)


def outer_bounds(quotes, window, amin, amax, cells=16, progress=QUIET):
    amin,amax = rational(amin),rational(amax)
    if not 0 < amin <= amax: raise ValueError("Require 0 < amin <= amax")
    if cells < 2 or cells > 1024: raise ValueError("cells must be in [2,1024]")
    w = window
    pt,pk = bump(w.t,w.ht),bump(w.k,w.hk)
    nt = [-2*x for x in deriv(pt)]
    dk = deriv(deriv(mul([F(0),F(0),F(1)],pk)))
    # Include quote dates/strikes to exploit irregular observation geometry.
    ts = sorted({w.t-w.ht+2*w.ht*j/cells for j in range(cells+1)} |
                {q.t for q in quotes if w.t-w.ht < q.t < w.t+w.ht} | {w.t})
    ks = sorted({w.k-w.hk+2*w.hk*j/cells for j in range(cells+1)} |
                {q.k for q in quotes if w.k-w.hk < q.k < w.k+w.hk})
    n,d = I(0),I(0)
    done,total = 0,(len(ts)-1)*(len(ks)-1)
    progress.event("continuous_enclosure", "start",done=0,total=total)
    # Precompute t-dependent moment bounds and polynomial ranges via their
    # lru caches; all remaining arithmetic remains exact.
    for ti,(ta,tb) in enumerate(zip(ts[:-1],ts[1:])):
        for ka,kb in zip(ks[:-1],ks[1:]):
            try:
                c = price_envelope(quotes,ta,tb,ka,kb,amax)
            except ValueError:
                return dict(scope="all_bounded_variance_continuous_diffusions",
                            status="empty_by_price_envelope", empty_class=True,
                            assumptions=dict(amin=str(amin),amax=str(amax)),
                            window=w.json(), cells=cells, quotes=quote_payload(quotes))
            n = (n+weighted_cell(nt,pk,ta,tb,ka,kb,c)).rounded()
            d = (d+weighted_cell(pt,dk,ta,tb,ka,kb,c)).rounded()
            done += 1
        progress.event("continuous_enclosure", "running",done=done,total=total)
    d = I(max(F(0),d.lo),d.hi) if d.hi >= 0 else d
    lower,upper = amin,amax
    if d.hi > 0:
        lower = max(lower,n.lo/d.hi)
    if d.lo > 0:
        upper = min(upper,n.hi/d.lo)
    empty = d.hi <= 0 or n.hi < 0 or lower > upper
    status = ("empty_by_weak_enclosure" if empty else
              "positive_denominator_proved" if d.lo>0 else "denominator_unresolved")
    result = dict(scope="all_bounded_variance_continuous_diffusions", status=status,
                  empty_class=empty, assumptions=dict(amin=str(amin),amax=str(amax),
                  normalization="X0=1; dX=X*sqrt(a(t,X))*dW; fixed deterministic F,D",
                  quotes_are="exact normalized input intervals; European calls"),
                  window=w.json(),cells=cells,actual_cells=total,
                  numerator=n.json(),denominator=d.json(),
                  denominator_positive_proved=d.lo>0,
                  quadrature="exact signed polynomial integration with rational cell enclosures",
                  tail_error="0: compactly supported kernel, no truncated PDE",
                  sharpness_proved=False, attainment_proved=False,
                  quotes=quote_payload(quotes))
    if not empty:
        result.update(lower=str(lower),upper=str(upper),width=str(upper-lower),
                      approximate=[float(lower),float(upper)])
    progress.event("continuous_enclosure","done",done=done,total=total,
                   status_detail=status)
    return result


def nested_outer(quotes, window, amin, amax, levels, progress=QUIET):
    history=[]
    lower,upper=rational(amin),rational(amax)
    for cells in levels:
        with progress.phase('continuous_outer_level',cells=cells):
            r=outer_bounds(quotes,window,amin,amax,cells,progress)
        history.append(r)
        if r["empty_class"]:
            return dict(status="empty_class",history=history,empty_class=True)
        lower=max(lower,F(r["lower"]));upper=min(upper,F(r["upper"]))
        if lower>upper:
            return dict(status="empty_class",history=history,empty_class=True)
    return dict(scope="continuous_outer_intersection",empty_class=False,
                lower=str(lower),upper=str(upper),width=str(upper-lower),
                approximate=[float(lower),float(upper)], history=history,
                stopping_claim="Refinement reduces enclosure dependency; does not prove sharpness")
