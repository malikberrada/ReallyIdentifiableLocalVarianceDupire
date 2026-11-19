"""A constructive bounded local-variance diffusion fitting original SPXW books.

At the selected expiry S_T = B exp(h(Z)), with Z standard normal and h
increasing, continuous and piecewise affine. X_t=E[S_T|W_t]/E[S_T].
Its relative volatility is a positive weighted average of the slopes of h
 divided by sqrt(T). Pricing is analytic and verified by rational intervals.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import isfinite

from .interval import I, black_call, exp_point, log_point, normal_cdf
from .spxw_books import clip, endpoints, span

KNOTS = tuple(map(F, ('-2','-1','-1/2','0','1/2','1','2')))


def intercepts(knots, slopes):
    """Enforce h(0)=0 and continuity at every knot (0 is a knot)."""
    constants = [0]
    for i,x in enumerate(knots):
        constants.append(constants[-1]+(slopes[i]-slopes[i+1])*x)
    offset = constants[list(knots).index(0)]
    return [b-offset for b in constants]


@lru_cache(maxsize=8192)
def _phi_point(x):
    return normal_cdf(I(x))


def phi(x):
    return I(_phi_point(x.lo).lo, _phi_point(x.hi).hi)


def at(bound, shift=0):
    return I(0 if bound == '-inf' else 1) if isinstance(bound,str) else phi(bound-shift)


def price_quantile(model, pairs, t, amin, amax):
    if model['kind'] != 'gaussian_terminal_loglinear':
        raise ValueError('Unknown diffusion witness')
    knots = tuple(map(F,model['knots']))
    slopes = tuple(map(F,model['slopes']))
    med, d = F(model['median_spot']), F(model['discount'])
    if (len(slopes) != len(knots)+1 or list(knots) != sorted(set(knots)) or 0 not in knots
            or med <= 0 or d <= 0 or any(s <= 0 or not amin*t <= s*s <= amax*t for s in slopes)):
        raise ValueError('Invalid diffusion coefficients, rates, or knots')
    constants = intercepts(knots,slopes)
    breaks = ['-inf',*map(I,knots),'+inf']
    pieces = []
    for i,(s,b) in enumerate(zip(slopes,constants)):
        l,r = breaks[i:i+2]
        weight = med*exp_point(b+s*s/2)
        pieces.append((l,r,s,b,weight,at(l,s),at(r,s),at(l),at(r)))
    forward = sum((weight*(pr-pl) for _,_,_,_,weight,pl,pr,_,_ in pieces), I(0))
    rows = []
    for p in pairs:
        log_k = log_point(p.k/med)
        call = I(0)
        for l,r,s,b,weight,pl,pr,ql,qr in pieces:
            threshold = (log_k-b)/s
            if r != '+inf' and threshold.lo >= r.hi:
                continue
            if l != '-inf' and threshold.hi <= l.lo:
                gm, prob = pr-pl, qr-ql
            else:
                low,high = threshold.lo,threshold.hi
                if l != '-inf':
                    low,high=max(low,l.lo),max(high,l.hi)
                if r != '+inf':
                    low,high=min(low,r.lo),min(high,r.hi)
                cut = I(low,high).rounded(80)
                gm, prob = pr-phi(cut-s), qr-phi(cut)
            call += weight*gm-p.k*prob
        call = (d*call).rounded()
        put = (call+d*(p.k-forward)).rounded()
        rows.extend([(p.call,call),(p.put,put)])
    return d,forward,rows


def verify_model(model,pairs,t,amin,amax,discount=None):
    if model['kind'] == 'constant_variance':
        d, fwd, a = (F(model[k]) for k in ('discount','forward','variance'))
        if min(d,fwd) <= 0 or not amin <= a <= amax:
            raise ValueError('Invalid constant-variance witness')
        fw = I(fwd)
        rows = []
        for p in pairs:
            call = d*fwd*black_call(p.k/fwd,a*t)
            rows.extend([(p.call,call),(p.put,call+d*(p.k-fwd))])
    else:
        d,fw,rows = price_quantile(model,pairs,t,amin,amax)
    if discount and not F(discount[0]) <= d <= F(discount[1]):
        raise ValueError('Witness outside explicit discount bounds')
    proof = []
    for book,value in rows:
        if value.lo < book.bid or value.hi > book.ask:
            raise ValueError(f'Witness outside raw bid/ask for {book.id}')
        proof.append(dict(id=book.id,bid=str(book.bid),ask=str(book.ask),price=endpoints(value),
                          slack_lower=str(min(value.lo-book.bid,book.ask-value.hi))))
    return dict(verified=True,discount=str(d),forward=endpoints(fw),prepaid_forward=endpoints(d*fw),
                checked_contracts=len(proof),minimum_slack_points=str(min(F(row['slack_lower']) for row in proof)),
                raw_prices=proof)


def _decimal(x):
    return str(F(format(float(x),'.15g')))


def _search(pairs, t, amin, amax, poly, discount=None):
    """Binary64 proposes parameters; only verify_model can accept a candidate."""
    import numpy as np
    from scipy.optimize import least_squares, minimize
    from scipy.special import ndtr

    strikes = np.array([float(p.k) for p in pairs])
    n = len(pairs)
    low = np.array([float(q.bid) for side in ('call','put') for p in pairs for q in [getattr(p,side)]])
    high = np.array([float(q.ask) for side in ('call','put') for p in pairs for q in [getattr(p,side)]])
    mid = (low+high)/2
    half=np.maximum((high-low)/2,1e-9)
    disc = span(poly, lambda d,a:d)
    f_range=span(poly,lambda d,a:a/d)
    d0=float((disc.lo+disc.hi)/2)
    f0=float((f_range.lo+f_range.hi)/2)
    lo_s, hi_s = np.sqrt([float(amin*t), float(amax*t)])
    near = int(np.argmin(abs(strikes-f0)))
    implied = (mid[near]+mid[n+near])/(2*d0*f0)*np.sqrt(2*np.pi)
    # Optimize D on [0,1] inside its identified range, including a fixed D.
    def D(u):
        return float(disc.lo)+(float(disc.hi-disc.lo))*u
    def raw_prices(x,knots):
        level=f0*np.exp(x[0])
        d=D(x[1])
        if knots is None:
            sig=x[2]
            d1=np.log(level/strikes)/sig+sig/2
            C=level*ndtr(d1)-strikes*ndtr(d1-sig)
            fw=level
        else:
            slopes=x[2:]
            b=np.array(intercepts(knots,slopes))
            br=np.r_[-np.inf,knots,np.inf]
            G=level*np.exp(b+slopes**2/2)
            th=np.clip((np.log(strikes[:,None]/level)-b)/slopes,br[:-1],br[1:])
            C=(G*(ndtr(br[1:]-slopes)-ndtr(th-slopes))-strikes[:,None]*(ndtr(br[1:])-ndtr(th))).sum(axis=1)
            fw=np.sum(G*(ndtr(br[1:]-slopes)-ndtr(br[:-1]-slopes)))
        return d*np.r_[C,C+strikes-fw]
    def try_model(x,knots):
        if not all(isfinite(v) for v in x):
            return None
        d = min(disc.hi,max(disc.lo,F(_decimal(D(x[1])))))
        level = _decimal(f0*np.exp(x[0]))
        if knots is None:
            rate = max(amin,min(amax,F(_decimal(x[2]**2/float(t)))))
            model=dict(kind='constant_variance',discount=str(d),forward=level,variance=str(rate))
        else:
            model=dict(kind='gaussian_terminal_loglinear',discount=str(d),median_spot=level,
                       knots=list(map(str,KNOTS)),slopes=[_decimal(v) for v in x[2:]])
        try:
            verify_model(model,pairs,t,amin,amax,discount)
        except (ValueError,ArithmeticError):
            return None
        return model
    for knots in (None,np.array(list(map(float,KNOTS)))):
        count=1 if knots is None else len(knots)+1
        if amin == amax and knots is not None:
            break
        # Padding only permits SciPy to search an exactly fixed variance in binary64.
        width=max(1e-14,(hi_s-lo_s)*1e-10)
        sl=lo_s+width if amin<amax else max(1e-14,lo_s-width)
        sh=hi_s-width if amin<amax else hi_s+width
        if sh <= sl:
            sl,sh=max(1e-14,lo_s-width),hi_s+width
        bounds=(np.r_[-min(12.,5*hi_s+1),0,np.full(count,sl)],np.r_[min(12.,5*hi_s+1),1,np.full(count,sh)])
        x0=np.r_[0.,.5,np.full(count,np.clip(implied,sl,sh))]
        def residual(x):
            return (raw_prices(x,knots)-mid)/half
        def objective(x):
            r=residual(x)
            return np.sign(r)*np.maximum(abs(r)-.80,0.)+1e-5*r
        fit=least_squares(objective,x0,bounds=bounds,max_nfev=350,ftol=1e-12,xtol=1e-12,gtol=1e-12)
        if np.max(abs(residual(fit.x))) <= 1.0000001:
            model=try_model(fit.x,knots)
            if model is not None:
                return model
        # Minimax refinement avoids a least-squares compromise outside a narrow spread.
        if knots is not None:
            def constraints(v):
                r=residual(v[:-1])
                return np.r_[v[-1]-r,v[-1]+r]
            v0=np.r_[fit.x,np.max(abs(residual(fit.x)))+1e-6]
            opt=minimize(lambda v:v[-1],v0,method='SLSQP',bounds=[*zip(*bounds),(0,None)],
                         constraints=[{'type':'ineq','fun':constraints}],options={'maxiter':300,'ftol':2e-11})
            if np.max(abs(residual(opt.x[:-1]))) <= 1.0000001:
                model=try_model(opt.x[:-1],knots)
                if model is not None:
                    return model
    return None


def propose(pairs,t,amin,amax,poly,discount=None):
    # Try an interior parity discount first. It is only a witness choice;
    # the outer problem always uses the entire identified polygon.
    ds=span(poly,lambda d,a:d)
    if ds.lo != ds.hi:
        middle=(ds.lo+ds.hi)/2
        section=clip(clip(poly,lambda x:x[0]-middle),lambda x:middle-x[0])
        model=_search(pairs,t,amin,amax,section,discount)
        if model is not None:
            return model
    model=_search(pairs,t,amin,amax,poly,discount)
    if model is None:
        raise ValueError('No common witness verified in the searched families; general feasibility is undetermined')
    return model
