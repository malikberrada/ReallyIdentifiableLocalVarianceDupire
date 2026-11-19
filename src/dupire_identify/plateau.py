"""Fixed diffusions with an exact maximum-variance rectangle.

The numerical PDE is only a proposal. A piecewise bicubic Hermite surface is
checked by outward Bernstein residuals. The parabolic comparison principle
then encloses its true continuous option prices, including the entire
positive half line via bounded-volatility boundary comparisons.
"""
from fractions import Fraction as F
from pathlib import Path
import json
import math
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded
from scipy.special import ndtr
from . import binary_intervals as bi
from .binary_intervals import B
from .interval import I, black_call, normal_cdf, log_point, sqrt_point
from .io import rational, quote_payload, payload_quotes, save_json, sha256, read_quotes, write_quotes
from .progress import QUIET, Progress
from .continuous import Window
from . import denominator, time_models

try:
    from ._native import tridiagonal as _native_tridiagonal
except ImportError:
    _native_tridiagonal=None


def proposal_engine():return 'C++ Thomas' if _native_tridiagonal is not None else 'LAPACK through SciPy'


def band_solve(matrix,rhs):
    return _native_tridiagonal(matrix,rhs) if _native_tridiagonal is not None else solve_banded((1,1),matrix,rhs)


def coefficient(config, phase, strikes):
    """C2 in space, piecewise constant in time; convex quintic interpolation."""
    knots=np.array(list(map(float,map(F,config['strike_knots']))))
    controls=np.array(list(map(float,map(F,config['phases'][phase]['variance']))))
    x=np.asarray(strikes);j=np.clip(np.searchsorted(knots,x,side='right')-1,0,len(knots)-2)
    u=np.clip((x-knots[j])/(knots[j+1]-knots[j]),0,1)
    s=u*u*u*(10-15*u+6*u*u)
    return controls[j]*(1-s)+controls[j+1]*s


def validate_config(config):
    if config.get('kind')!='piecewise_quintic_plateau_diffusion_v1':raise ValueError('Unknown diffusion format')
    lo,hi=map(F,(config['amin'],config['amax']));a0=F(config['prefix_variance'])
    t0=F(config['prefix_end']);knots=list(map(F,config['strike_knots']))
    if not 0<lo<=a0<=hi or t0<=0 or len(knots)<2 or knots[0]<=0 or any(a>=b for a,b in zip(knots,knots[1:])):
        raise ValueError('Invalid plateau diffusion')
    last=t0
    for p,phase in enumerate(config['phases']):
        if F(phase['left'])!=last or F(phase['right'])<=last:raise ValueError('Invalid time partition')
        vals=list(map(F,phase['variance']))
        pk=phase_knots(config,p)
        if len(vals)!=len(pk) or pk[0]!=knots[0] or pk[-1]!=knots[-1] or any(a>=b for a,b in zip(pk,pk[1:])) or min(vals)<lo or max(vals)>hi:
            raise ValueError('Variance control outside prior or invalid phase grid')
        last=F(phase['right'])
    return lo,hi


def default_config(quotes, window):
    """Reproducible parameterization for the supplied synthetic input.

    No quote is silently replaced by synthetic data. A different input may
    require optimization or another parameterization and can remain open.
    """
    w=window;dates=sorted({q.t for q in quotes})
    if w!=Window.make('3/8',1,'1/16','1/10') or dates!=[F(1,4),F(3,4),F(1)]:
        raise ValueError('Supply --config for another date/window geometry')
    # Rounded dyadic candidates; all prices still require independent checks.
    x=[.0025,.0025,.09317673,.05246775,.03715291,.02666746,.04193695,.04229642,.09898902,
       .03394616,.03990839,.03977897,.04034625,.03881627,.04348347,.03123110]
    knots=list(map(F,['1/8','13/20','4/5','179/200','9/10','1','11/10','221/200','6/5','7/5','8']))
    a3=np.r_[x[2],x[2:9],x[8]];a4=np.r_[x[9],x[9:16],x[15]]
    # The extra .895/.9 and 1.1/1.105 panels are needed only in the pulse.
    # Each phase therefore carries its own panel knots.
    config=dict(kind='piecewise_quintic_plateau_diffusion_v1',amin='1/400',amax='1/4',
        prefix_end='1/4',prefix_variance='1/25',strike_knots=list(map(str,knots)),phases=[])
    def add(l,r,k,a):
        config['phases'].append(dict(left=l,right=r,knots=list(map(str,k)),variance=[str(F(float(v)).limit_denominator(2**32)) for v in a]))
    add('1/4','5/16',knots,[F(1,400)]*len(knots))
    add('5/16','7/16',knots,[F(1,400)]*4+[F(1,4)]*3+[F(1,400)]*4)
    other=list(map(F,['1/8','13/20','4/5','9/10','1','11/10','6/5','7/5','8']))
    add('7/16','3/4',other,a3);add('3/4','1',other,a4)
    return config


def phase_knots(config,phase):return list(map(F,config['phases'][phase].get('knots',config['strike_knots'])))


def a_values(config,phase,strikes):
    c=dict(config,strike_knots=list(map(str,phase_knots(config,phase))))
    return coefficient(c,phase,strikes)


def variance_envelope(config,time):
    """Deterministic integrated-variance comparisons specific to the model."""
    time=F(time);prefix=F(config['prefix_end']);a0=F(config['prefix_variance'])
    lower=upper=a0*min(time,prefix)
    for phase in config['phases']:
        length=max(F(0),min(time,F(phase['right']))-F(phase['left']))
        values=list(map(F,phase['variance']));lower+=length*min(values);upper+=length*max(values)
    last=F(config['phases'][-1]['right'])
    if time>last:lower+=a0*(time-last);upper+=a0*(time-last)
    return lower,upper


def make_grid(config, spatial_cells=384, pulse_cells=64):
    """Exact rational grid, with local refinement of coefficient transitions."""
    all_knots=set()
    for p in range(len(config['phases'])):all_knots.update(phase_knots(config,p))
    original=sorted(all_knots)
    pulse_bands=[(a-F(1,40),b+F(1,40),b-a) for a,b in zip(original,original[1:]) if b-a<=F(1,100)]
    for a,b,width in pulse_bands:all_knots.update([max(original[0],a),min(original[-1],b)])
    knots=sorted(all_knots);grid=set(knots)
    for a,b in zip(knots,knots[1:]):
        # Most gamma lives near 1. The tails get larger cells; all errors are
        # checked, so this heuristic never supplies an accuracy claim.
        mid=(a+b)/2
        density=spatial_cells if F(1,2)<mid<F(3,2) else max(32,spatial_cells//4)
        n=max(1,math.ceil(float((b-a)*density)))
        if b-a<=F(1,100):n=max(n,pulse_cells)
        for lo,hi,width in pulse_bands:
            if lo<=a and b<=hi:n=max(n,math.ceil(float((b-a)/width*pulse_cells)))
        grid.update(a+(b-a)*j/n for j in range(n+1))
    # Keep compact-spline stencils from crossing abrupt mesh-size changes.
    while True:
        ordered=sorted(grid);added=[]
        for a,b,c in zip(ordered,ordered[1:],ordered[2:]):
            if b-a>2*(c-b):added.append((a+b)/2)
            if c-b>2*(b-a):added.append((b+c)/2)
        if not added:break
        grid.update(added)
    return sorted(grid)


def make_times(left,right,steps=1024):
    """Small steps after a coefficient jump, increasing to 1/steps."""
    left,right=F(left),F(right);h=F(1,steps*1024);cap=F(1,steps);t=left;times=[t];count=0
    while t<right:
        t=min(right,t+h);times.append(t);count+=1
        if count%6==0:h=min(cap,2*h)
    return times


def compact_operator(grid):
    h=np.diff(grid);n=len(grid)-2
    # R M = Q C for the natural cubic spline.
    rlo=h[:-1]/6;rdiag=(h[:-1]+h[1:])/3;rhi=h[1:]/6
    qlo=1/h[:-1];qdiag=-(1/h[:-1]+1/h[1:]);qhi=1/h[1:]
    return rlo,rdiag,rhi,qlo,qdiag,qhi


def solve_numerical(config,grid,time_steps=1024,progress=QUIET):
    """Cubic compact spatial proposal, L-stable SDIRK2 time steps.

    No claimed convergence rate is used in the certificate.
    """
    validate_config(config);x=np.array(list(map(float,grid)));n=len(x)
    v=float(F(config['prefix_end'])*F(config['prefix_variance']));s=np.sqrt(v)
    d1=(-np.log(x)+v/2)/s
    c=ndtr(d1)-x*ndtr(d1-s);c[0]=1-x[0];c[-1]=0.
    op=compact_operator(x);rl,rd,ru,ql,qd,qu=op
    R=np.zeros((3,n-2));R[0,1:]=ru[:-1];R[1]=rd;R[2,:-1]=rl[1:]
    def apply_q(z):return ql*z[:-2]+qd*z[1:-1]+qu*z[2:]
    arrays={};phase_times=[]
    for p,phase in enumerate(config['phases']):
        times=make_times(phase['left'],phase['right'],time_steps);phase_times.append(list(map(str,times)))
        a=a_values(config,p,x);D=.5*a*x*x
        states=[]
        for it,t in enumerate(times):
            M=np.r_[0.,band_solve(R,apply_q(c)),0.]
            cs=CubicSpline(x,c,bc_type='natural');dt=D*M
            ds=CubicSpline(x,dt,bc_type='natural')
            sx=cs(x,1)
            if p==0 and it==0:sx=-ndtr(d1-s)
            states.append(np.stack([c.copy(),sx,dt,ds(x,1)]))
            if it+1==len(times):break
            h=float(times[it+1]-t)
            gamma=1-1/np.sqrt(2.)
            A=R.copy();A[1]-=gamma*h*qd*D[1:-1]
            A[0,1:]-=gamma*h*qu[:-1]*D[2:-1]
            A[2,:-1]-=gamma*h*ql[1:]*D[1:-2]
            m1=band_solve(A,apply_q(c))
            z=c+(1-gamma)*h*np.r_[0.,D[1:-1]*m1,0.]
            m2=band_solve(A,apply_q(z))
            c=z+gamma*h*np.r_[0.,D[1:-1]*m2,0.]
            if it%max(1,len(times)//24)==0:
                progress.event('plateau_pde_proposal','running',done=it+1,total=len(times)-1,phase=p+1)
        arrays[f'phase_{p}']=np.asarray(states)
        progress.event('plateau_pde_proposal','done',phase=p+1,steps=len(times)-1)
    return arrays,phase_times


def coefficient_bernstein(config,phase,grid):
    """Coefficient enclosures on complete cells, retaining exact rational data."""
    knots=phase_knots(config,phase);vals=list(map(F,config['phases'][phase]['variance']))
    lows=[];highs=[]
    for ka,kb in zip(grid,grid[1:]):
        j=next(j for j in range(len(knots)-1) if knots[j]<=ka and kb<=knots[j+1])
        coeff=bi.stack([B.rational(v) for v in [vals[j]]*3+[vals[j+1]]*3],axis=-1)
        u=B.rational((ka-knots[j])/(knots[j+1]-knots[j]));v=B.rational((kb-knots[j])/(knots[j+1]-knots[j]))
        restricted=bi.restrict(coeff,u,v)
        lows.append(restricted.lo);highs.append(restricted.hi)
    a=B(np.asarray(lows),np.asarray(highs))
    ka=B(np.array([float(q) for q in grid[:-1]]));kb=B(np.array([float(q) for q in grid[1:]]))
    # Grid nodes need not be exactly representable.
    ka=bi.stack([B.rational(q) for q in grid[:-1]],axis=0)
    kb=bi.stack([B.rational(q) for q in grid[1:]],axis=0)
    k2=bi.stack([ka*ka,ka*kb,kb*kb],axis=-1)
    return bi.product(a,k2)/2


def residual_bounds(config,grid,phase_times,arrays,progress=QUIET,batch_size=12):
    """Exact-candidate residual bounds; array arithmetic rounded outwards."""
    bi.self_check();validate_config(config)
    if len(grid)<3 or any(a>=b for a,b in zip(grid,grid[1:])):raise ValueError('Invalid spatial partition')
    if len(phase_times)!=len(config['phases']):raise ValueError('Missing time phase')
    required=set()
    for p in range(len(phase_times)):required.update(phase_knots(config,p))
    if not required<=set(grid) or grid[0]!=min(required) or grid[-1]!=max(required):
        raise ValueError('PDE mesh must cover and subdivide every coefficient panel')
    dx=bi.stack([B.rational(b-a) for a,b in zip(grid,grid[1:])],axis=0)
    results=[]
    for phase,times_text in enumerate(phase_times):
        times=list(map(F,times_text));z=arrays[f'phase_{phase}'];n=len(times)-1
        spec=config['phases'][phase]
        if n<1 or times[0]!=F(spec['left']) or times[-1]!=F(spec['right']) or any(a>=b for a,b in zip(times,times[1:])):
            raise ValueError('PDE time mesh differs from the specified diffusion')
        if z.shape!=(len(times),4,len(grid)) or np.any(~np.isfinite(z)):
            raise ValueError('Invalid bicubic payload dimensions or nonfinite coefficients')
        if phase and not np.array_equal(z[0,:2],arrays[f'phase_{phase-1}'][-1,:2]):
            raise ValueError('Hermite surface is discontinuous at a phase join')
        # Equal boundary values and zero time slopes imply constant boundaries.
        for side,want in ((0,1-float(grid[0])),(-1,0.)):
            if not np.all(z[:,0,side]==want) or not np.all(z[:,2,side]==0):
                raise ValueError('Incorrect PDE boundary polynomial')
        ak2=coefficient_bernstein(config,phase,grid)
        rho=[]
        for start in range(0,n,batch_size):
            stop=min(n,start+batch_size)
            h=bi.stack([B.rational(times[i+1]-times[i]) for i in range(start,stop)],axis=0)
            def spatial(values,slopes):
                val=B(values);slope=B(slopes)
                return bi.stack([val[...,:-1],val[...,:-1]+dx*slope[...,:-1]/3,
                                 val[...,1:]-dx*slope[...,1:]/3,val[...,1:]],axis=-1)
            left=spatial(z[start:stop,0],z[start:stop,1]);right=spatial(z[start+1:stop+1,0],z[start+1:stop+1,1])
            lt=spatial(z[start:stop,2],z[start:stop,3]);rt=spatial(z[start+1:stop+1,2],z[start+1:stop+1,3])
            # Tensor axes: time cells, space cells, time degree, space degree.
            v=bi.stack([left,left+h[:,None,None]*lt/3,right-h[:,None,None]*rt/3,right],axis=-2)
            vt=3*(v[...,1:,:]-v[...,:-1,:])/h[:,None,None,None]
            # Elevate time degree 2 -> 3 explicitly (spatial axis stays last).
            vt=bi.stack([vt[...,0,:],(vt[...,0,:]+2*vt[...,1,:])/3,
                         (2*vt[...,1,:]+vt[...,2,:])/3,vt[...,2,:]],axis=-2)
            vkk=6*(v[...,2:]-2*v[...,1:-1]+v[...,:-2])/(dx[None,:,None,None]*dx[None,:,None,None])
            rhs=bi.product(vkk,ak2[None,:,None,:])
            residual=bi.elevate(vt,8)-rhs
            vals=residual.absolute_upper().max(axis=(1,2,3))
            rho.extend(map(float,vals))
            where=np.unravel_index(np.argmax(residual.absolute_upper()),residual.lo.shape)
            progress.event('plateau_residual_bernstein','running',done=stop,total=n,phase=phase+1,
                           residual_upper=float(vals.max()),worst_k=str(grid[where[1]]))
        results.append(rho)
    return results


def initial_error(config,grid,first,progress=QUIET):
    """Hermite interpolation error from rational Black endpoint enclosures."""
    v=F(config['prefix_end'])*F(config['prefix_variance']);s=sqrt_point(v)
    prices=[];slopes=[]
    left_tail=min(grid,key=lambda x:abs(x-F(13,20)))
    right_tail=min(grid,key=lambda x:abs(x-F(7,5)))
    left_budget=black_call(left_tail,v).hi-(1-left_tail)
    right_budget=black_call(right_tail,v).hi
    for i,k in enumerate(grid):
        if k<left_tail or k>right_tail:
            prices.append(None);slopes.append(None)
            continue
        p=black_call(k,v);d2=(-log_point(k)-v/2)/s
        slope=-normal_cdf(d2)
        prices.append(p);slopes.append(slope)
        if i%max(1,len(grid)//24)==0:progress.event('plateau_initial_black','running',done=i+1,total=len(grid))
    worst=F(0)
    progress.event('plateau_initial_black','done',done=len(grid),total=len(grid))
    from .density import gamma_box
    for i,(a,b) in enumerate(zip(grid,grid[1:])):
        if b<=left_tail or a>=right_tail:
            slope=F(-1) if b<=left_tail else F(0)
            budget=left_budget if b<=left_tail else right_budget
            e=max(abs(F(float(first[0,j]))-(1-grid[j] if b<=left_tail else 0)) for j in (i,i+1))
            de=max(abs(F(float(first[1,j]))-slope) for j in (i,i+1))
            worst=max(worst,e+(b-a)*de/4+budget)
            continue
        e=max(abs(F(float(first[0,j]))-p) for j in (i,i+1) for p in (prices[j].lo,prices[j].hi))
        de=max(abs(F(float(first[1,j]))-p) for j in (i,i+1) for p in (slopes[j].lo,slopes[j].hi))
        # C'''' = C''/k^2 * ((3/2+log(k)/v)^2+(3/2+log(k)/v)-1/v).
        ell=I(log_point(a).lo,log_point(b).hi);beta=F(3,2)+ell/v
        shape=beta*beta+beta-1/v
        g=gamma_box(F(config['prefix_end']),F(config['prefix_end']),a,b,F(config['prefix_variance']))[0]
        fourth=g.hi/a**4*max(abs(shape.lo),abs(shape.hi))
        worst=max(worst,e+(b-a)*de/4+(b-a)**4*fourth/384)
        if i%max(1,len(grid)//24)==0:
            progress.event('plateau_initial_error_cells','running',done=i+1,total=len(grid)-1)
    progress.event('plateau_initial_error_cells','done',done=len(grid)-1,total=len(grid)-1)
    return worst


def price_value(grid,times,z,t,k):
    """Rational evaluation of the candidate at a requested quote."""
    def hermite(v0,v1,s0,s1,width,u):
        return (2*u**3-3*u*u+1)*v0+(-2*u**3+3*u*u)*v1+(u**3-2*u*u+u)*width*s0+(u**3-u*u)*width*s1
    it=next(i for i in range(len(times)-1) if times[i]<=t<=times[i+1])
    ik=next(i for i in range(len(grid)-1) if grid[i]<=k<=grid[i+1])
    dx=grid[ik+1]-grid[ik];u=(k-grid[ik])/dx
    fields=[]
    for j in (it,it+1):
        vals=[F(float(z[j,field,l])) for field in range(4) for l in (ik,ik+1)]
        fields.append((hermite(vals[0],vals[1],vals[2],vals[3],dx,u),
                       hermite(vals[4],vals[5],vals[6],vals[7],dx,u)))
    width=times[it+1]-times[it];u=(t-times[it])/width
    return hermite(fields[0][0],fields[1][0],fields[0][1],fields[1][1],width,u)


def certify_prices(config,grid,phase_times,arrays,quotes,progress=QUIET):
    lo,hi=validate_config(config)
    if any(q.t>F(config['phases'][-1]['right']) or not grid[0]<=q.k<=grid[-1] for q in quotes):
        raise ValueError('Quote outside certified horizon/domain')
    rho=residual_bounds(config,grid,phase_times,arrays,progress)
    initial=initial_error(config,grid,arrays['phase_0'][0],progress)
    t0=F(config['prefix_end']);a0=F(config['prefix_variance']);reports=[];cumulative=F(0)
    for q in quotes:
        if q.t<=t0:
            price=black_call(q.k,a0*q.t);eta=F(0);value=None
        else:
            cumulative=F(0);phase_index=None
            for p,times_text in enumerate(phase_times):
                times=list(map(F,times_text))
                for i,(ta,tb) in enumerate(zip(times,times[1:])):
                    if ta<q.t:cumulative+=(min(q.t,tb)-ta)*F(rho[p][i])
                if times[0]<=q.t<=times[-1] and phase_index is None:phase_index=p
            p=phase_index;times=list(map(F,phase_times[p]))
            v=variance_envelope(config,q.t)[1] if config.get('price_envelope_intersection',False) else a0*t0+hi*(q.t-t0)
            # Uniform boundary discrepancy up to this quote's maturity.
            left_offset=abs(F(float(arrays['phase_0'][0,0,0]))-(1-grid[0]))
            boundary=max(black_call(grid[0],v).hi-(1-grid[0])+left_offset,black_call(grid[-1],v).hi)
            eta=max(initial,boundary)+cumulative
            value=price_value(grid,times,arrays[f'phase_{p}'],q.t,q.k)
            price=I(value-eta,value+eta)
            if config.get('price_envelope_intersection',False):
                # This is a proven comparison bound, not truncation of an
                # unfavourable residual or silent clipping to a quote spread.
                low_variance,high_variance=variance_envelope(config,q.t)
                price=price.intersect(I(black_call(q.k,low_variance).lo,black_call(q.k,high_variance).hi))
        reports.append(dict(id=q.id,price=price.json(),eta=str(eta),candidate_value=str(value) if value is not None else None,
            within_bid_ask=price.lo>=q.bid and price.hi<=q.ask))
        progress.event('plateau_quote_compatibility','running',done=len(reports),total=len(quotes),
                       eta=float(eta),compatible=reports[-1]['within_bid_ask'])
    return dict(all_quotes_compatible=all(r['within_bid_ask'] for r in reports),prices=reports,
        initial_error=str(initial),residual_integrals=[str(sum((b-a)*F(v) for a,b,v in zip(map(F,ts[:-1]),map(F,ts[1:]),rs))) for ts,rs in zip(phase_times,rho)],
        arithmetic='outward binary64 Bernstein intervals; rational initial, boundary and quote evaluation',
        theorem='|C-v| <= max(initial error,boundary comparison)+integral supremum PDE residual',
        domain=[str(grid[0]),str(grid[-1])],exterior='bounded-variance Black comparison; no absorbing diffusion boundary imposed')


def plateau_rectangles(config,target):
    rectangles=[]
    for p,phase in enumerate(config['phases']):
        knots=phase_knots(config,p);vals=list(map(F,phase['variance']));start=None
        for j,(a,b) in enumerate(zip(knots,knots[1:])):
            match=vals[j]==vals[j+1]==target
            if match and start is None:start=a
            if start is not None and (not match or j==len(knots)-2):
                end=a if not match else b
                rectangles.append(dict(t_left=phase['left'],t_right=phase['right'],k_left=str(start),k_right=str(end),variance=str(target)))
                start=None
    return rectangles
