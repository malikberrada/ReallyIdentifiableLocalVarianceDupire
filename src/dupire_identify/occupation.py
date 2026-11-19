"""Continuous outer relaxation through probability/variance occupation measures.

No absorbing/reflecting boundary is imposed. The final state cell is [R,+inf),
with probability, first and second moment variables. Every bounded-variance
continuous model projects into the rational inequalities assembled here.
"""
from fractions import Fraction as F
from functools import lru_cache
import numpy as np
from .certlp import RationalLP,fractional_outer
from .interval import exp_point,black_call,log_point
from .polynomial import bump,bernstein_range
from .curves import test_curve,moment_enclosure
from .io import rational
from .density import gamma_box
from .progress import QUIET


def addto(row,j,v):
    if v:row[j]=row.get(j,F(0))+v


@lru_cache(maxsize=8192)
def moment_upper(t,amax,power):
    return exp_point(F(power*(power-1),2)*amax*t).hi


def cell_upper(t,amax,left,order):
    bound=F(1) if order<2 else moment_upper(t,amax,2)
    if left>0:
        for p in range(max(2,order),13):
            if F(p*(p-1),2)*amax*t>90:break
            bound=min(bound,moment_upper(t,amax,p)/left**(p-order))
    return bound


def grids(quotes,w,time_cells,strike_cells,kmax,target_cells=8):
    horizon=max(max(q.t for q in quotes),w.t+w.ht)
    times=sorted({F(0),horizon,w.t-w.ht,w.t,w.t+w.ht} |
                 {w.t-w.ht+2*w.ht*i/target_cells for i in range(target_cells+1)} |
                 {horizon*i/time_cells for i in range(time_cells+1)} |
                 {q.t for q in quotes})
    right=max(rational(kmax),max(q.k for q in quotes)*F(5,4),w.k+w.hk*2)
    # Concentrate spatial cells around the observed/target region; retain tails.
    lower=max(F(1,100),min(min(q.k for q in quotes),w.k-w.hk)/2)
    upper=min(right,max(max(q.k for q in quotes),w.k+w.hk)*F(5,4))
    strikes=sorted({F(0),F(1),right,lower,upper,w.k-w.hk,w.k,w.k+w.hk} |
                   {q.k for q in quotes} |
                   {lower+(upper-lower)*i/strike_cells for i in range(strike_cells+1)} |
                   {w.k-w.hk+2*w.hk*i/target_cells for i in range(target_cells+1)})
    return times,strikes


def assemble(quotes,w,amin,amax,time_cells=8,strike_cells=16,kmax=3,
             time_degree=1,bs_bounds=True,target_cells=8,comparison_cuts=True,progress=QUIET,
             denominator_floor=None,window_family=None):
    amin,amax=map(rational,(amin,amax))
    if not 0<amin<amax or time_degree not in (1,2) or time_cells<2 or strike_cells<4:
        raise ValueError('Invalid occupation relaxation settings')
    if target_cells<2:raise ValueError('At least two target cells required')
    if window_family is not None:
        from .families import Family
        if not isinstance(window_family,Family):window_family=Family.make(window_family)
        if w!=window_family.cover_window():raise ValueError('The assembly grid must cover the complete kernel family')
    t,k=grids(quotes,w,time_cells,strike_cells,kmax,target_cells)
    J=len(k);lp=RationalLP();terminal={};occ={};targets=[]
    # Terminal variables (P,S,H) = integrals of (1,x,x²) over each state cell.
    def moment_rows(triple,left,right,label):
        p,s,h=triple
        lp.row({p:left,s:-1},0,label+'_first_lower')
        if right is not None:
            lp.row({s:1,p:-right},0,label+'_first_upper')
            lp.row({h:1,s:-(left+right),p:left*right},0,label+'_second_upper')
            anchors=(left,(left+right)/2,right)
        else:anchors=tuple(left*2**i for i in range(7))
        # Exact linear supporting cuts of S²<=P*H, valid for every measure.
        for z in anchors:
            lp.row({h:-1,s:2*z,p:-z*z},0,label+'_moment_cut')
    for it,tt in enumerate(t[1:],1):
        for j,left in enumerate(k):
            triple=tuple(lp.var(cell_upper(tt,amax,left,r),f'term_{it}_{j}_{r}') for r in range(3))
            terminal[it,j]=triple
            moment_rows(triple,left,k[j+1] if j+1<J else None,'terminal')
        for r in (0,1):lp.eq({terminal[it,j][r]:1 for j in range(J)},1,'terminal_moment')
        lp.row({terminal[it,j][2]:1 for j in range(J)},moment_upper(tt,amax,2),'second_moment_upper')
        lp.row({terminal[it,j][2]:-1 for j in range(J)},-exp_point(amin*tt).lo,'second_moment_lower')
    pt,pk=bump(w.t,w.ht),bump(w.k,w.hk)
    for it,(ta,tb) in enumerate(zip(t[:-1],t[1:])):
        dt=tb-ta
        for r in range(time_degree+1):
            for j,left in enumerate(k):
                unit=dt/F(time_degree+1)
                m=[lp.var(unit*cell_upper(tb,amax,left,o),f'occ_{it}_{r}_{j}_{o}') for o in range(3)]
                flux=lp.var(amax*unit*cell_upper(tb,amax,left,2),f'flux_{it}_{r}_{j}')
                occ[it,r,j]=(*m,flux)
                moment_rows(m,left,k[j+1] if j+1<J else None,'occupation')
                lp.row({m[2]:amin,flux:-1},0,'variance_lower')
                lp.row({flux:1,m[2]:-amax},0,'variance_upper')
                if (j+1<J and ta>=w.t-w.ht and tb<=w.t+w.ht and
                    left>=w.k-w.hk and k[j+1]<=w.k+w.hk):
                    weight=(window_family.weights(ta,tb,left,k[j+1]) if window_family is not None else
                            bernstein_range(pt,ta,tb)*bernstein_range(pk,left,k[j+1]))
                    wl,wu=max(F(0),weight.lo),max(F(0),weight.hi)
                    z=lp.var(wu*lp.upper[m[2]],f'weighted_mass_{it}_{r}_{j}')
                    y=lp.var(amax*lp.upper[z],f'weighted_flux_{it}_{r}_{j}')
                    targets.append((y,z))
                    lp.row({m[2]:wl,z:-1},0,'kernel_mass_lower')
                    lp.row({z:1,m[2]:-wu},0,'kernel_mass_upper')
                    lp.row({flux:wl,y:-1},0,'kernel_flux_lower')
                    lp.row({y:1,flux:-wu},0,'kernel_flux_upper')
                    lp.row({z:amin,y:-1},0,'kernel_variance_lower')
                    lp.row({y:1,z:-amax},0,'kernel_variance_upper')
            for order in (0,1):
                lp.eq({occ[it,r,j][order]:1 for j in range(J)},unit,'occupation_moment')
            lp.row({occ[it,r,j][2]:1 for j in range(J)},unit*moment_upper(tb,amax,2),'occupation_global_second_upper')
            lp.row({occ[it,r,j][2]:-1 for j in range(J)},-unit*exp_point(amin*ta).lo,'occupation_global_second_lower')
        # Exact x² Dynkin identities, including the unbounded right tail.
        for affine in (False,True):
            row={}
            if affine:
                for j in range(J):addto(row,terminal[it+1,j][2],dt)
                for r in range(time_degree+1):
                    for j in range(J):
                        addto(row,occ[it,r,j][2],-1)
                        addto(row,occ[it,r,j][3],-dt*F(r,time_degree))
                lp.eq(row,0,'dynkin_x2_time')
            else:
                for j in range(J):
                    addto(row,terminal[it+1,j][2],1)
                    if it:addto(row,terminal[it,j][2],-1)
                    for r in range(time_degree+1):addto(row,occ[it,r,j][3],-1)
                lp.eq(row,1 if it==0 else 0,'dynkin_x2')
    # Calls are evaluated EXACTLY at all partition strikes, including quote K.
    def call_row(it,strike):
        row={}
        for j,left in enumerate(k):
            if left>=strike:
                p,s,_=terminal[it,j];addto(row,s,1);addto(row,p,-strike)
        return row
    for quote in quotes:
        row=call_row(t.index(quote.t),quote.k)
        lp.row(row,quote.ask,'observed_ask')
        lp.row({j:-v for j,v in row.items()},-quote.bid,'observed_bid')
    for strike in k[1:]:
        prev=None
        for it,tt in enumerate(t[1:],1):
            row=call_row(it,strike)
            if prev is not None:
                diff=dict(prev)
                for j,v in row.items():addto(diff,j,-v)
                lp.row(diff,0,'call_calendar')
            prev=row
    # Global bounded-volatility comparison. Use original quote strikes + target
    # boundaries, rather than spending arithmetic on every auxiliary node.
    if bs_bounds:
        selected=sorted({q.k for q in quotes}|{w.k-w.hk,w.k,w.k+w.hk})
        total=len(selected)*(len(t)-1);done=0
        with progress.phase('continuous_black_comparison',prices=total):
            for it,tt in enumerate(t[1:],1):
                for strike in selected:
                    ell=log_point(strike)
                    distance=F(0) if ell.lo<=0<=ell.hi else min(abs(ell.lo),abs(ell.hi))
                    # Intrinsic is always a valid lower bound; avoid expensive
                    # tiny Gaussian tails that add no useful data precision.
                    lo=max(F(0),1-strike) if distance*distance>16*amin*tt else black_call(strike,amin*tt).lo
                    hi=black_call(strike,amax*tt).hi
                    row=call_row(it,strike)
                    lp.row(row,hi,'global_vol_call_upper')
                    lp.row({j:-v for j,v in row.items()},-lo,'global_vol_call_lower')
                    done+=1
                progress.event('continuous_black_comparison','running',done=done,total=total)
    # Smooth call and digital test functions couple occupation densities/flux
    # and the terminal laws. The quadratic interpolation remainder is enclosed
    # on each complete cell by exact Bernstein coefficients.
    curves=[test_curve(k[j-1],k[j+1],kind) for j in range(1,J-1) for kind in ('call','digital')]
    for ci,curve in enumerate(curves):
        bounds=[moment_enclosure(curve,left,k[j+1] if j+1<J else None) for j,left in enumerate(k)]
        seconds=[]
        for j,left in enumerate(k):
            right=k[j+1] if j+1<J else None
            polynomial=curve.piece(left,right,2)
            br=bernstein_range(polynomial,left,right) if right is not None else None
            seconds.append((br.lo,br.hi) if br else (F(0),F(0)))
        def terminal_expr(it,upper):
            if it==0:return {},curve.at(F(1))
            row={}
            for j,(coef,errlo,errhi) in enumerate(bounds):
                a,b,c=coef;a+=errhi if upper else errlo
                for variable,v in zip(terminal[it,j],(a,b,c)):addto(row,variable,v)
            return row,F(0)
        for it,(ta,tb) in enumerate(zip(t[:-1],t[1:])):
            dt=tb-ta
            # A bounded expression L<=exact identity=0<=U yields two valid rows.
            for kind in ('constant','right_time','left_time'):
                for upper in (False,True):
                    row={};constant=F(0)
                    def term_terminal(index,mult):
                        nonlocal constant
                        rr,con=terminal_expr(index,upper if mult>0 else not upper)
                        for j,v in rr.items():addto(row,j,mult*v)
                        constant+=mult*con
                    if kind=='constant':term_terminal(it+1,1);term_terminal(it,-1)
                    elif kind=='right_time':term_terminal(it+1,dt)
                    else:term_terminal(it,dt)
                    for r in range(time_degree+1):
                        for j,(coef,errlo,errhi) in enumerate(bounds):
                            mass,first,second,flux=occ[it,r,j]
                            if kind!='constant':
                                a,b,c=coef;a+=errlo if upper else errhi
                                for variable,v in zip((mass,first,second),(a,b,c)):addto(row,variable,-v)
                            factor=F(-1,2) if kind=='constant' else (
                                -dt*F(r,2*time_degree) if kind=='right_time' else dt*F(time_degree-r,2*time_degree))
                            lo,hi=seconds[j]
                            coefficient=(hi if upper else lo) if factor>=0 else (lo if upper else hi)
                            addto(row,flux,factor*coefficient)
                    # L<=0 ; -U<=0.
                    sign=-1 if upper else 1
                    lp.row({j:sign*v for j,v in row.items()},-sign*constant,'weak_diffusion_'+kind)
        progress.event('continuous_weak_tests','running',done=ci+1,total=len(curves))
    n=[F(0)]*len(lp.upper);d=n.copy()
    for y,z in targets:n[y]=1;d[z]=1
    comparisons=[]
    if comparison_cuts:
        eligible=[q for q in quotes if q.t>w.t+w.ht]
        for index,quote in enumerate(eligible):
            for reference,side in ((amin,'lower_reference'),(amax,'upper_reference')):
                dominance=F(0);valid=True
                for it in range(4):
                    ta=w.t-w.ht+2*w.ht*it/4;tb=ta+2*w.ht/4
                    tr=bernstein_range(pt,ta,tb)
                    for jk in range(4):
                        ka=w.k-w.hk+2*w.hk*jk/4;kb=ka+2*w.hk/4
                        kr=bernstein_range(pk,ka,kb)
                        phi=(window_family.weights(ta,tb,ka,kb).hi if window_family is not None else max(F(0),(tr*kr).hi))
                        g=gamma_box(quote.t-tb,quote.t-ta,ka/quote.k,kb/quote.k,reference)[0]
                        # Backward Black gamma = K*G(remaining_time,x/K)/x².
                        gamma_lower=quote.k*g.lo/(kb*kb)
                        if gamma_lower<=0:valid=False;break
                        dominance=max(dominance,phi/gamma_lower)
                    if not valid:break
                if not valid:continue
                call=black_call(quote.k,reference*quote.t)
                budget=(quote.ask-call.lo) if side=='lower_reference' else (call.hi-quote.bid)
                if budget<0:raise ValueError('Quote excludes the bounded-variance class: '+quote.id)
                rhs=2*dominance*budget
                if side=='lower_reference':row={j:n[j]-amin*d[j] for j in range(len(n)) if n[j] or d[j]}
                else:row={j:amax*d[j]-n[j] for j in range(len(n)) if n[j] or d[j]}
                lp.row(row,rhs,'ito_black_flux_budget')
                comparisons.append(dict(quote=quote.id,reference=str(reference),side=side,
                    gamma_domination=str(dominance),price_budget=str(budget),rhs=str(rhs)))
            progress.event('continuous_flux_comparisons','running',done=index+1,total=len(eligible))
    definition=dict(time_cells=time_cells,strike_cells=strike_cells,kmax=str(kmax),
                    time_degree=time_degree,bs_bounds=bs_bounds,target_cells=target_cells,times=list(map(str,t)),strikes=list(map(str,k)),
                    comparison_cuts=comparison_cuts,flux_budget_cuts=comparisons,
                    right_tail='[last_strike,+infinity), retained probability/first/second moments',
                    variables=len(lp.upper),constraints=len(lp.rows),tests=len(curves))
    if window_family is not None:definition['window_family']=window_family.json()
    if denominator_floor is not None:
        floor=rational(denominator_floor)
        if not F(1,2**900)<=floor<=F(1,2) or floor.denominator&(floor.denominator-1) or floor.numerator!=1:
            raise ValueError('The linked continuous denominator floor must be a representable positive dyadic power <= 1/2')
        index=len(lp.rows)
        lp.row({j:-v for j,v in enumerate(d) if v},-floor,'verified_continuous_denominator_floor')
        definition.update(denominator_floor=str(floor),denominator_row=index,constraints=len(lp.rows))
    return lp,n,d,definition


def bounds(quotes,w,amin,amax,directory,progress=QUIET,**settings):
    time_limit=settings.pop('time_limit',120)
    with progress.phase('occupation_relaxation_assembly'):
        lp,n,d,definition=assemble(quotes,w,amin,amax,progress=progress,**settings)
    result=fractional_outer(lp,n,d,directory,progress,time_limit,definition.get('denominator_row'))
    lower=max(rational(amin),rational(result.get('lower',amin)))
    upper=min(rational(amax),rational(result.get('upper',amax)))
    if lower>upper:raise ArithmeticError('Empty or inconsistent continuous relaxation')
    return dict(**result,scope='all_bounded_variance_continuous_diffusions',
                definition=definition,continuous_lower=str(lower),continuous_upper=str(upper),
                width=str(upper-lower),projection_verified=True,
                proof='Weak Dynkin identities, moment inequalities, interval polynomial ranges; outward LP rounding')
