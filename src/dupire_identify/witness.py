"""Exact continuous witnesses in a deterministic-variance subclass.

The common quote prices are closed-form Black prices. No PDE tail,
time-discretization or statistical Monte Carlo error is used in this branch.
"""
from fractions import Fraction as F
from .interval import I, black_call, pi, sin_interval_lipschitz
from .io import rational, quote_payload
from .continuous import Window
from .progress import QUIET


def choose_gap(quotes):
    dates = sorted({q.t for q in quotes})
    if len(dates)<2: raise ValueError("At least two observed maturities required")
    return max(zip(dates[:-1],dates[1:]),key=lambda x:x[1]-x[0])


def witness_pair(quotes, a0="1/25", epsilon="1/50", amin="1/400", amax="1/4",
                 left=None, right=None, window=None, progress=QUIET):
    a0,e,amin,amax=map(rational,(a0,epsilon,amin,amax))
    if left is None or right is None: left,right=choose_gap(quotes)
    else: left,right=map(rational,(left,right))
    if left<0 or right<=left or e<=0 or not 0<amin<=a0-e<=a0+e<=amax:
        raise ValueError("Witness variance/gap inconsistent with announced assumptions")
    if any(left<q.t<right for q in quotes):
        raise ValueError("An observed maturity lies strictly inside the proposed gap")
    gap=right-left
    w=window or Window.make(left+gap/4,1,gap/8,"1/10")
    compatibility=[]
    for i,q in enumerate(quotes):
        price=black_call(q.k,a0*q.t)
        accepted=q.bid<=price.lo and price.hi<=q.ask
        compatibility.append(dict(id=q.id,price=price.json(),within_bid_ask=accepted,
                                  eta=str((price.hi-price.lo)/2)))
        progress.event("analytic_witness_prices","running",done=i+1,total=len(quotes))
    compatible=all(row["within_bid_ask"] for row in compatibility)
    # On a positive sine lobe its minimum is the minimum of the endpoints.
    # The universal maximum is 1. Model-dependent density weights are positive.
    positive_lobe=left <= w.t-w.ht < w.t+w.ht <= left+gap/2
    separated=False
    minus=I(a0-e,a0+e);plus=minus
    ming=F(0)
    if positive_lobe:
        gs=[sin_interval_lipschitz(2*pi()*(t-left)/gap)
            for t in (w.t-w.ht,w.t+w.ht)]
        ming=max(F(0),min(g.lo for g in gs))
        minus=I(a0-e,a0-e*ming)
        plus=I(a0+e*ming,a0+e)
        separated=ming>0
    gap_lower=max(F(0),plus.lo-minus.hi) if separated else F(0)
    certified=compatible and separated
    return dict(scope="continuous_analytic_deterministic_variance_subclass",
                status="continuous_witness_pair_verified" if certified else
                       "incompatible_quotes" if not compatible else "window_not_separated",
                compatible=compatible,separated=separated,certified=certified,
                a0=str(a0),epsilon=str(e),amin=str(amin),amax=str(amax),
                left=str(left),right=str(right),window=w.json(),
                model="a_plus/minus(t,k)=a0 +/- epsilon*sin(2*pi*(t-left)/(right-left)) inside gap; a0 outside",
                observed_integrated_variance="Exactly a0*T for both models (symbolic zero integral)",
                minus_average=minus.json(),plus_average=plus.json(),
                separation_lower=str(gap_lower),minimax_error_lower=str(gap_lower/2),
                common_prices=compatibility,quotes=quote_payload(quotes),
                pricing_errors=dict(discretization="0",truncation="0",optimization="0",
                                    arithmetic="rational interval series with explicit remainders"),
                endpoint_attainment_of_full_class_proved=False)
