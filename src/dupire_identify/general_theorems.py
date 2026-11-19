"""Quote-driven universal sufficient regions, with explicit hypotheses.

The small-window result uses a verified Lipschitz diffusion and strict quote
slack, not a Black smile. It is a coupling certificate, not a claim that every
longer window has the full prior identification set.
"""
from fractions import Fraction as F
from . import plateau
from .interval import sqrt_point,I,exp_point
from .io import quote_payload


def power_compare(value,coefficient,negative_exponent):
    """value <= coefficient*2**(-n), without allocating astronomical ints."""
    value,coefficient=F(value),F(coefficient);n=int(negative_exponent)
    if value<=0:return True
    if coefficient<=0 or n<0:raise ValueError('Positive coefficient and nonnegative exponent required')
    ratio=value/coefficient
    # A difference of at least 2 in binary lengths decides most comparisons.
    estimate=ratio.numerator.bit_length()-ratio.denominator.bit_length()+n
    if estimate>=2:return False
    if estimate<=-2:return True
    if n>ratio.denominator.bit_length()+2:return False
    return ratio.numerator*(1<<n)<=ratio.denominator


def quote_slack(quotes,pricing):
    if [p['id'] for p in pricing['prices']]!=[q.id for q in quotes]:raise ValueError('Pricing order differs from quotes')
    return min(min(F(p['price']['lo'])-q.bid,q.ask-F(p['price']['hi'])) for q,p in zip(quotes,pricing['prices']))


def regions(config,quotes,pricing):
    amin,amax=plateau.validate_config(config)
    if not pricing['all_quotes_compatible']:return None
    m=quote_slack(quotes,pricing);root_min=sqrt_point(amin);root_max=sqrt_point(amax)
    root_lower=max(root_min.lo,min(amin,F(1)))
    lip=root_max.hi
    for phase in range(len(config['phases'])):
        ks=plateau.phase_knots(config,phase);values=list(map(F,config['phases'][phase]['variance']))
        for a,b,v0,v1 in zip(ks,ks[1:],values,values[1:]):
            lip=max(lip,root_max.hi+b*F(15,8)*abs(v1-v0)/(2*root_lower*(b-a)))
    difference=(root_max.hi-root_lower)**2
    horizon=max(q.t for q in quotes)
    # In one dimension the transition semigroup of the base martingale is
    # 1-Lipschitz: synchronous ordered solutions stay ordered and preserve
    # their expected separation. There is no exp(L_sigma^2*T) amplification.
    energy=amax*horizon
    v=F(3,2)*energy;exponent=(v.numerator+v.denominator-1)//v.denominator
    width=None
    if m>0:
        if energy<=100:
            growth=exp_point(energy).hi;coefficient=m*m/(16*amax*growth);power=0
        else:
            growth=None;coefficient=m*m/(16*amax);power=-exponent
        width=dict(coefficient=str(coefficient),binary_exponent=power,
            condition='2*hT <= coefficient*2**binary_exponent',all_time_centers=True,all_positive_spatial_supports=True)
    return dict(kind='quote_slack_coupling_regions_v1',quotes=quote_payload(quotes),
        amin=str(amin),amax=str(amax),strict_slack=str(m),global_sigma_lipschitz_upper=str(lip),
        square_root_variance_difference_squared_upper=str(difference),maximum_observed_maturity=str(horizon),
        exp_energy_upper=str(energy),exp_upper_power_two=exponent,
        full_prior_interval=[str(amin),str(amax)],
        small_time_width=width,
        future=dict(left_support_at_least=str(horizon),all_positive_spatial_supports=True),
        hypotheses='Verified compatible fixed diffusion, global state Lipschitz coefficient; strict price slack for the small-window clause',
        proof='1D martingale transition semigroup is 1-Lipschitz; triangle inequality; Ito isometry; E[X^2]<=exp(amax*T); call payoff is 1-Lipschitz',
        price_change_bound='2*sqrt(amax*exp(amax*T_last)*window_time_width), at most half the certified quote slack',
        modification='Keep the base coefficient except on the entire chosen time window, where it is the desired prior endpoint for all x>0',
        all_positive_parameters_solved=False)


def covers(family,doc):
    if doc is None:return None
    if family.support()[0]>=F(doc['future']['left_support_at_least']):
        return dict(kind='unobserved_future',interval=doc['full_prior_interval'])
    w=doc['small_time_width']
    if w and power_compare(2*family.intervals[2][1],F(w['coefficient']),-int(w['binary_exponent'])):
        return dict(kind='strict_slack_small_time',interval=doc['full_prior_interval'])
    return None


def endpoint_config(config,window,target):
    """Construct the actual coefficient prescribed by the coupling theorem."""
    import json
    target=F(target);amin,amax=plateau.validate_config(config)
    if target not in (amin,amax):raise ValueError('A prior endpoint is required')
    out=json.loads(json.dumps(config));old_prefix=F(config['prefix_end']);old_end=F(config['phases'][-1]['right'])
    left,right=window.t-window.ht,window.t+window.ht
    start=min(old_prefix,left)/2;end=max(old_end,right)
    points=sorted({start,old_prefix,left,right,end}|{F(p[k]) for p in config['phases'] for k in ('left','right')})
    out['prefix_end']=str(start);out['phases']=[]
    for a,b in zip(points,points[1:]):
        if a<start or b>end:continue
        mid=(a+b)/2
        if left<=mid<=right:
            knobs=config['strike_knots'];values=[str(target)]*len(knobs)
        elif mid<old_prefix or mid>old_end:
            knobs=config['strike_knots'];values=[config['prefix_variance']]*len(knobs)
        else:
            phase=next(p for p in config['phases'] if F(p['left'])<=mid<=F(p['right']))
            knobs=phase.get('knots',config['strike_knots']);values=phase['variance']
        out['phases'].append(dict(left=str(a),right=str(b),knots=list(knobs),variance=list(values)))
    plateau.validate_config(out)
    return out


def quick_infeasibility(quotes,amin,amax,vol_checks=True):
    """Exact sufficient contradictions; absence is not proof of feasibility."""
    from .interval import black_call
    for i,a in enumerate(quotes):
        for b in quotes[i+1:]:
            if a.t==b.t:
                low,high=sorted((a,b),key=lambda q:q.k)
                if low.ask<high.bid:return dict(kind='call_monotonicity',ids=[low.id,high.id])
                if low.bid-high.ask>high.k-low.k:return dict(kind='vertical_spread',ids=[low.id,high.id])
            if a.k==b.k:
                early,late=sorted((a,b),key=lambda q:q.t)
                if early.bid>late.ask:return dict(kind='calendar',ids=[early.id,late.id])
    for maturity in sorted({q.t for q in quotes}):
        group=sorted((q for q in quotes if q.t==maturity),key=lambda q:q.k)
        for a,b,c in zip(group,group[1:],group[2:]):
            if c.k==a.k:continue
            weight=(c.k-b.k)/(c.k-a.k)
            if b.bid>weight*a.ask+(1-weight)*c.ask:return dict(kind='butterfly_convexity',ids=[a.id,b.id,c.id])
    if vol_checks:
        for q in quotes:
            low=black_call(q.k,F(amin)*q.t);high=black_call(q.k,F(amax)*q.t)
            if q.ask<low.lo:return dict(kind='lower_variance_prior',ids=[q.id])
            if q.bid>high.hi:return dict(kind='upper_variance_prior',ids=[q.id])
    return None
