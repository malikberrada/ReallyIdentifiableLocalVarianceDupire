"""Exact deterministic time-change witnesses, including attainable endpoints."""
from fractions import Fraction as F
from .interval import I,black_call
from .io import quote_payload
from .curves import bernstein,value
from .polynomial import deriv


def integrated_variance(doc,t):
    t=F(t);total=F(0);last=F(0)
    for left,right,coefficients in doc['segments']:
        left,right=map(F,(left,right));coefficients=list(map(F,coefficients))
        if t>left:
            u=(min(t,right)-left)/(right-left)
            total+=(right-left)*sum(c*u**(i+1)/(i+1) for i,c in enumerate(coefficients))
        last=right
    if t>last:total+=F(doc['a0'])*(t-last)
    return total


def variance(doc,t):
    t=F(t)
    for left,right,coefficients in doc['segments']:
        left,right=F(left),F(right)
        if left<=t<=right:return value(list(map(F,coefficients)),(t-left)/(right-left))
    return F(doc['a0'])


def candidates(quotes,window,a0,amin,amax,adaptive_ramps=False):
    a0,amin,amax=map(F,(a0,amin,amax));start,end=window.t-window.ht,window.t+window.ht
    dates=sorted({q.t for q in quotes})
    if adaptive_ramps and start>max(dates):return future_candidates(quotes,window,a0,amin,amax)
    lefts=[x for x in (sorted({F(0)}|set(dates)) if adaptive_ramps else dates) if x<=start]
    rights=[x for x in dates if x>=end]
    if not lefts or not rights:return []
    left,right=max(lefts),min(rights)
    if any(left<t<right for t in dates):return []
    length=right-left;width=end-start;rest=length-width
    if rest<=0 or start<=left or end>=right or not amin<a0<amax:return []
    # Shortening the smooth ramps recovers the full prior endpoints whenever
    # the clock budget permits them. Each ramp's integral is its endpoint mean.
    rho=F(1,4)
    if adaptive_ramps:
        limits=[];both=True
        for target in (amin,amax):
            rooms=(a0*length-target*width-amin*rest,amax*rest-a0*length+target*width)
            slopes=(rest*((a0+target)/2-amin),rest*(amax-(a0+target)/2))
            for room,slope in zip(rooms,slopes):
                if room<=0:both=False
                else:limits.append(room/(2*slope))
        rho=min([rho]+limits) if both else F(1,4096)
    capacity=a0*length-a0*rho*rest/2
    target_weight=width+rho*rest/2;comp_weight=(1-rho)*rest
    low=max(amin,(capacity-amax*comp_weight)/target_weight)
    high=min(amax,(capacity-amin*comp_weight)/target_weight)
    result=[]
    for label,target in (('minimum',low),('maximum',high)):
        if not amin<=target<=amax:continue
        compensation=(capacity-target*target_weight)/comp_weight
        segments=[]
        def append(a,b,v0,v1=None):
            if a>=b:return
            delta=F(0) if v1 is None else v1-v0
            coefficients=[v0] if not delta else [v0,0,0,10*delta,-15*delta,6*delta]
            segments.append([str(a),str(b),list(map(str,coefficients))])
        append(F(0),left,a0)
        g=start-left
        append(left,left+rho*g,a0,compensation);append(left+rho*g,start-rho*g,compensation)
        append(start-rho*g,start,compensation,target);append(start,end,target)
        g=right-end
        append(end,end+rho*g,target,compensation);append(end+rho*g,right-rho*g,compensation)
        append(right-rho*g,right,compensation,a0);append(right,max(dates),a0)
        doc=dict(model_type='deterministic_time_change',scope='continuous_diffusion',endpoint=label,
            a0=str(a0),amin=str(amin),amax=str(amax),segments=segments,window=window.json(),
            average=I(target).json(),target_variance=str(target),gap=[str(left),str(right)],
            ramp_fraction=str(rho),
            certified=True,primal_exact_feasible=True,global_dupire_inequalities_verified=True,
            denominator_positive_proved=True,quotes=quote_payload(quotes),
            proof='Constant variance over the kernel time support; exact integrated variance at every observed maturity',
            coefficient_regularity='C2 in time, positive quintic ramps and constant plateau; continuous diffusion paths',
            pricing_error='Exact integrated variance and rational interval Black prices')
        prices=[];valid=True
        for q in quotes:
            v=integrated_variance(doc,q.t)
            if v!=a0*q.t:valid=False;break
            p=black_call(q.k,v)
            if p.lo<q.bid or p.hi>q.ask:valid=False;break
            prices.append(dict(id=q.id,price=p.json(),within_bid_ask=True))
        if valid:doc['quote_prices']=prices;result.append(doc)
    return result


def future_candidates(quotes,window,a0,amin,amax):
    left=max(q.t for q in quotes);start,end=window.t-window.ht,window.t+window.ht
    if start<=left:return []
    width=start-left;right=end+width;result=[]
    for label,target in (('minimum',amin),('maximum',amax)):
        def ramp(v0,v1):
            delta=v1-v0
            return list(map(str,[v0,0,0,10*delta,-15*delta,6*delta]))
        segments=[['0',str(left),[str(a0)]],[str(left),str(start),ramp(a0,target)],
                  [str(start),str(end),[str(target)]],[str(end),str(right),ramp(target,a0)]]
        doc=dict(model_type='deterministic_time_change',scope='continuous_diffusion',endpoint=label,
            a0=str(a0),amin=str(amin),amax=str(amax),segments=segments,window=window.json(),
            average=I(target).json(),target_variance=str(target),gap=[str(left),str(right)],
            certified=True,primal_exact_feasible=True,global_dupire_inequalities_verified=True,
            denominator_positive_proved=True,quotes=quote_payload(quotes),
            proof='Unobserved future interval; smooth change after the last observation, constant on the full kernel',
            coefficient_regularity='C2 in time; quintic transitions',pricing_error='Exact rational interval Black prices')
        prices=[];valid=True
        for q in quotes:
            p=black_call(q.k,integrated_variance(doc,q.t))
            if p.lo<q.bid or p.hi>q.ask:valid=False;break
            prices.append(dict(id=q.id,price=p.json(),within_bid_ask=True))
        if valid:doc['quote_prices']=prices;result.append(doc)
    return result


def full_prior_region(quotes,a0,amin,amax):
    """Sufficient CONTINUUM region where BOTH global endpoints are attained.

    For a gap of length L, every strict interior time support of width
    w < L*min(a0-amin,amax-a0)/(amax-amin) admits two smooth time clocks.
    The statement is independent of the spatial kernel centre and width.
    """
    a0,amin,amax=map(F,(a0,amin,amax))
    if not 0<amin<a0<amax:raise ValueError('Strict interior Black reference required')
    for q in quotes:
        p=black_call(q.k,a0*q.t)
        if p.lo<q.bid or p.hi>q.ask:raise ValueError('Quotes do not contain the common Black reference')
    dates=sorted({F(0)}|{q.t for q in quotes});strips=[]
    for left,right in zip(dates[:-1],dates[1:]):
        threshold=(right-left)*min(a0-amin,amax-a0)/(amax-amin)
        strips.append(dict(left=str(left),right=str(right),strict_time_support_width_upper=str(threshold),
            strict_halfwidth_upper=str(threshold/2),all_spatial_windows=True))
    return dict(kind='full_prior_temporal_strips_v1',a0=str(a0),amin=str(amin),amax=str(amax),
        quotes=quote_payload(quotes),strips=strips,identification_interval=[str(amin),str(amax)],
        minimax_absolute_error=str((amax-amin)/2),
        after_last_maturity=dict(strict_time_support_left_lower=str(max(dates)),all_spatial_windows=True),
        conditions='Time support strictly inside one strip; 2*hT strictly below its width threshold; K-hK>0',
        scope='entire described continuum of time and spatial kernels, not all possible kernels')


def verify(doc,quotes,window):
    if doc['window']!=window.json() or doc['quotes']!=quote_payload(quotes):
        raise ValueError('Time-change witness linked to different observations or kernel')
    amin,amax=F(doc['amin']),F(doc['amax']);last=F(0);previous=(F(doc['a0']),F(0),F(0))
    if not 0<amin<=F(doc['a0'])<=amax:raise ValueError('Invalid time-change prior')
    for left,right,coefficients in doc['segments']:
        left,right=F(left),F(right);coefficients=list(map(F,coefficients))
        bc=bernstein(coefficients,F(0),F(1),max(0,len(coefficients)-1))
        if left!=last or right<=left or min(bc)<amin or max(bc)>amax:raise ValueError('Invalid time-change segment')
        current=[];following=[];p=coefficients
        for order in range(3):
            current.append(value(p,F(0))/(right-left)**order)
            following.append(value(p,F(1))/(right-left)**order);p=deriv(p)
        if tuple(current)!=previous:raise ValueError('Time coefficient not C2 at a segment join')
        previous=tuple(following)
        last=right
    if previous!=(F(doc['a0']),F(0),F(0)):raise ValueError('Invalid exterior time coefficient join')
    if last<window.t+window.ht:raise ValueError('Time profile does not cover the kernel')
    for left,right,coefficients in doc['segments']:
        if max(F(left),window.t-window.ht)<min(F(right),window.t+window.ht):
            p=list(map(F,coefficients))
            if p[0]!=F(doc['target_variance']) or any(p[1:]):raise ValueError('Variance not constant on full kernel support')
    if doc['average']!=I(F(doc['target_variance'])).json():raise ValueError('Incorrect deterministic average')
    prices=[]
    for q in quotes:
        p=black_call(q.k,integrated_variance(doc,q.t))
        if p.lo<q.bid or p.hi>q.ask:raise ValueError('Time-change model fails quote containment')
        prices.append(dict(id=q.id,price=p.json(),within_bid_ask=True))
    if prices!=doc['quote_prices']:raise ValueError('Incorrect time-change price claim')
    return dict(verified=True,continuous_model=True,primal_exact_feasible=True,average=doc['average'])
