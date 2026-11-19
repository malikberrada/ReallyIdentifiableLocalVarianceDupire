"""Uniform interval for the gamma of the LISTED strike under every parity curve."""
from fractions import Fraction as F
from math import ceil, floor

from .interval import I, black_call, exp_point, log_point, pi, sqrt
from .spxw_books import clip, endpoints, span
from .spxw_subhedge import Subhedges


def normalized(poly,p):
    k=span(poly,lambda d,a:p.k*d/a)
    cb=span(poly,lambda d,a:p.call.bid/a).lo
    ca=span(poly,lambda d,a:p.call.ask/a).hi
    pb=span(poly,lambda d,a:p.put.bid/a).lo
    pa=span(poly,lambda d,a:p.put.ask/a).hi
    # Put-call parity and natural bounds in normalized units.
    c=I(max(cb,pb+1-k.hi,F(0)),min(ca,pa+1-k.lo,F(1)))
    put=I(max(pb,cb+k.lo-1,F(0)),min(pa,ca+k.hi-1,k.hi))
    return k,c,put


def choose_reference(poly,p,t,amin,amax):
    d=sum(v[0] for v in poly)/len(poly)
    a=sum(v[1] for v in poly)/len(poly)
    k=p.k*d/a
    bid=max(p.call.bid/a,p.put.bid/a+1-k)
    ask=min(p.call.ask/a,p.put.ask/a+1-k)
    target=(bid+ask)/2
    l,h=amin,amax
    for _ in range(19):
        m=(l+h)/2
        price=black_call(k,m*t)
        if price.hi < target:
            l=m
        elif price.lo > target:
            h=m
        else:
            return m
    return (l+h)/2


def price_gap(poly,p,t,rate):
    # D slices maintain dependence between A, D, the strike and the numeraire.
    ds=span(poly,lambda d,a:d)
    bounds=[]
    for i in range(8 if ds.lo!=ds.hi else 1):
        left=ds.lo+(ds.hi-ds.lo)*i/8
        right=ds.lo+(ds.hi-ds.lo)*(i+1)/8
        part=clip(clip(poly,lambda v:v[0]-left),lambda v:right-v[0])
        k,c,put=normalized(part,p)
        # Both European Black prices are monotone in strike, in opposite directions.
        b_left=black_call(k.lo,rate*t)
        b_right=black_call(k.hi,rate*t)
        delta=I(max(c.lo-b_left.hi,put.lo-(b_right.hi+k.hi-1)),
                min(c.hi-b_right.lo,put.hi-(b_left.lo+k.lo-1)))
        bounds.append(delta)
    return I(min(x.lo for x in bounds),max(x.hi for x in bounds))


def tail_bounds(quotes,left,right):
    # Convex order from time t to T and positive payoffs on the two tail events.
    pl=pr=F(1)
    for k,c,p in quotes:
        if k.lo>left:
            pl=min(pl,p.hi/(k.lo-left))
        if k.hi<right:
            pr=min(pr,c.hi/(right-k.hi))
    return pl,pr


def masses(quotes,k,t,rate,amax,corridor,positions=None,steps=72,bands=40):
    lo,hi=corridor if corridor else (None,None)
    if corridor and not 0 < lo < k.lo <= k.hi < hi:
        raise ValueError('The corridor must contain every possible normalized target strike')
    hedger=Subhedges(quotes,k,t,rate,corridor)
    if positions is not None and len(positions)!=steps:
        raise ValueError("Wrong number of static hedge positions")
    base=2*pi()
    band_data=[]
    # Nest around the full interval of possible target strikes. Clipping to
    # a common fixed corridor preserves nesting and the probability inequality.
    center=(k.lo+k.hi)/2
    for j in range(1,bands+1):
        w=center*F(j,2*bands)
        left,right=center-w,center+w
        if corridor:
            left,right=max(lo,left),min(hi,right)
        dist=max(abs(log_point(left/k.hi).lo),abs(log_point(right/k.lo).hi))
        tails=tail_bounds(quotes,left,right)
        d0=min(1-left,right-1)
        band_data.append((sqrt(k.lo*left).lo,dist**2,max(F(0),1-sum(tails)),d0))
    if corridor:
        ext=[(lo,log_point(k.lo/lo).lo,True),(hi,log_point(hi/k.hi).lo,False)]
        outside_probs=tail_bounds(quotes,lo,hi)
    mass_lo=mass_up=out_hi=F(0)
    slabs=[]
    for i in range(steps):
        start=t*i/steps
        end=t*(i+1)/steps
        v0=rate*(t-end)
        v1=rate*(t-start)
        full=2*k.hi*(sqrt(v1).hi-sqrt(v0).lo)/(rate*sqrt(base).lo)
        mass_up+=full
        low=F(0)
        proposal={'slope':'0','short':{}}
        if v0>0:
            denom=sqrt(base*v1).hi
            # E[(X_t-1)^2] <= exp(amax*t)-1. If outside the exp engine's
            # range the quote-based probability lower bound still applies.
            moment=exp_point(amax*end).hi-1 if amax*end <= 100 else None
            heights=[]
            probs=[]
            for factor,dist2,inside,offset in band_data:
                e=F(ceil(64*(dist2/(2*v0)+v1/8)),64)
                height=F(0) if e>100 else I(factor*exp_point(-e).lo/denom).rounded(64).lo
                heights.append(min(height,heights[-1]) if heights else height)
                prob=max(inside,1-moment/offset**2) if moment is not None and offset>0 else inside
                probs.append(prob)
            low=I(sum((a-b)*p for a,b,p in zip(heights,heights[1:]+[F(0)],probs))).rounded(64).lo
            heights=hedger.heights(v0,v1)
            proposal=hedger.propose(heights) if positions is None else positions[i]
            low=max(low,hedger.check(proposal,heights))
            mass_lo+=(end-start)*low
        tail=F(0)
        if corridor:
            for (edge,distance,is_left),prob in zip(ext,outside_probs):
                upper=full
                if distance>0 and (is_left or distance>=v1/2):
                    at=min(v1,distance**2)
                    scale=sqrt(base*at).lo
                    if scale>0:
                        energy=min(F(100),F(floor(64*distance**2/(2*at)),64))
                        height=sqrt(edge*k.hi).hi*exp_point(-energy).hi/scale
                        upper=min(full,(end-start)*height)
                tail+=prob*upper
        tail=I(min(full,tail)).rounded(80).hi
        out_hi+=tail
        slabs.append(dict(time=list(map(str,(start,end))),integrand_floor=str(low),tail_mass_upper=str(tail),subhedge=proposal))
    return I(mass_lo,mass_up).rounded(80),I(out_hi).rounded(80).hi,slabs


def rounded_estimate(bounds,eps,digits=6):
    scale=10**digits
    m=(bounds.lo+bounds.hi)/2
    r=(bounds.hi-bounds.lo)/2
    unit=round(m*scale)
    value=F(unit,scale)
    err=ceil(max(value-bounds.lo,bounds.hi-value)*scale)
    def fmt(u):
        w,d=divmod(abs(u),scale)
        return ('-' if u<0 else '')+str(w)+(f'.{d:0{digits}d}' if digits else '')
    return dict(outer=endpoints(bounds),midpoint=str(m),radius=str(r),epsilon=str(eps),
                display_outer=[fmt(floor(bounds.lo*scale)),fmt(ceil(bounds.hi*scale))],
                estimate=fmt(unit),absolute_error_at_most=fmt(err),exact_target_met=r<=eps,
                displayed_target_met=F(err,scale)<=eps)


def prove_gamma(poly,pairs,target,t,settings,positions=None):
    amin,amax,eps=map(F,(settings['amin'],settings['amax'],settings['epsilon']))
    requested=settings['reference']
    rate=choose_reference(poly,target,t,amin,amax) if requested=='auto' else F(requested)
    corridor=tuple(map(F,settings['corridor'])) if settings['corridor'] else None
    quotes=[normalized(poly,p) for p in pairs]
    k=quotes[pairs.index(target)][0]
    d,out,slabs=masses(quotes,k,t,rate,amax,corridor,positions)
    if d.lo<=0:
        raise ValueError('Could not prove a strictly positive denominator for this corridor')
    gap=price_gap(poly,target,t,rate)
    # 2(C-B_ref) = E int (a-ref) X² Gamma dt over the whole state space.
    numer=2*gap - I(min(F(0),amin-rate)*out,max(F(0),amax-rate)*out)
    result=(rate+numer/d).intersect(I(amin,amax)).rounded(80).intersect(I(amin,amax))
    return dict(call_id=target.call.id,listed_strike=str(target.k),normalized_strike_range=endpoints(k),
                reference_variance=str(rate),reference_selection=requested,corridor=settings['corridor'],
                call_minus_reference_price=endpoints(gap),denominator_bounds=endpoints(d),
                outside_gamma_mass_upper=str(out),time_slabs=slabs,
                estimate=rounded_estimate(result,eps,settings['digits']))
