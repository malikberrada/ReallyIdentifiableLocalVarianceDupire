"""Global denominator lower bounds from exactly verified Ito subsolutions.

V=P-epsilon*(H-t), with P a tensor compact C2 spline, satisfies
V_t + a*x*x*V_xx/2 + phi*x*x >= 0 for EVERY a in [amin,amax].
V(H,.)=0, so Ito implies D >= V(0,1). Numerical LP residuals are
absorbed by an exact global epsilon; no feasibility tolerance is trusted.
"""
from fractions import Fraction as F
from functools import lru_cache
import numpy as np
from scipy import sparse
from .continuous import Window
from .curves import spline_piece,bernstein
from .polynomial import bump,mul,deriv
from .inner import basis_values
from .native_solver import optimize
from .io import rational,quote_payload,payload_quotes
from .progress import QUIET
from .interval import log_point,pi,sqrt,exp_point


def dyadic_floor(x):
    x=F(x)
    if x<=0:return F(0)
    e=x.numerator.bit_length()-x.denominator.bit_length()
    q=F(2**e) if e>=0 else F(1,2**(-e))
    return q if q<=x else q/2


def brownian_floor(window,amin,amax):
    """Uniform positive floor via DDS and survival of drifted Brownian motion.

    X_t=exp(B_{A_t}-A_t/2), amin*t<=A_t<=amax*t. A single event that
    keeps B_u-u/2 inside a log band for ALL relevant clock times therefore
    controls every admissible stochastic clock, without independence claims.
    """
    amin,amax=map(rational,(amin,amax));w=window
    if not 0<amin<=amax:raise ValueError('Strictly positive lower variance required')
    if exp_point(F(2)).hi>=8:raise ArithmeticError('Exponential constant proof failed')
    best=None
    for rt in (F(1,4),F(1,2),F(3,4)):
        tlo,thi=w.t-rt*w.ht,w.t+rt*w.ht;s=amin*tlo;end=amax*thi
        for rk in (F(1,2),F(3,4),F(7,8),F(15,16)):
            left,right=w.k-rk*w.hk,w.k+rk*w.hk
            a,b=log_point(left).hi,log_point(right).lo;width=b-a
            if width<=0:continue
            entry_left=a+width/4;entry_right=b-width/4
            gaussian_energy=max(abs(entry_left+s/2),abs(entry_right+s/2))**2/(2*s)
            eigenvalue=pi().hi**2/(2*width*width)+F(1,8)
            energy=gaussian_energy+eigenvalue*(end-s)+(b-entry_left)/2
            # e^2<8 implies exp(-E) >= 2^(-ceil(3E/2)), E>=0.
            v=F(3,2)*energy;exponent=(v.numerator+v.denominator-1)//v.denominator
            pre=(thi-tlo)*left*left*(1-rt*rt)**3*(1-rk*rk)**3*width/(4*sqrt(2*pi()*s).hi)
            if exponent>12000:continue  # explicit arithmetic budget, never a fake zero floor
            lower=pre/F(2**exponent)
            if best is None or lower>F(best['lower_bound']):
                best=dict(kind='dds_survival_v1',window=w.json(),amin=str(amin),amax=str(amax),
                    time_radius=str(rt),strike_radius=str(rk),log_band=[str(a),str(b)],
                    earliest_clock=str(s),latest_clock=str(end),eigenvalue_upper=str(eigenvalue),
                    energy_upper=str(energy),power_two_exponent=exponent,prefactor=str(pre),
                    lower_bound=str(lower),dyadic_lower=str(min(F(1,2),dyadic_floor(lower))),
                    positive=True,approximate=float(lower),
                    scope='every continuous martingale with the announced strictly positive variance bounds',
                    proof='DDS; Gaussian density lower bound; killed Brownian eigenfunction; exp(2)<8')
    if best is None:raise ValueError('Kernel exceeds the validated log/arithmetic budget; no positivity claim generated')
    return best


def verify_brownian(doc,window=None,amin=None,amax=None):
    w=Window.make(**doc['window']);lo,hi=F(doc['amin']),F(doc['amax'])
    if window is not None and w!=window:raise ValueError('Different Brownian denominator kernel')
    if amin is not None and (lo!=F(amin) or hi!=F(amax)):raise ValueError('Different Brownian denominator class')
    if doc!=brownian_floor(w,lo,hi):raise ValueError('Incorrect Brownian denominator proof')
    return F(doc['dyadic_lower'])


def definition(window,nt=12,nk=16,start=F(0),quote_strikes=()):
    if nt<4 or nk<4:raise ValueError('At least four subsolution cells per axis')
    w=window;horizon=w.t+w.ht/2;step=(horizon-start)/nt
    local_t={w.t-w.ht+F(i,8)*2*w.ht for i in range(9)}
    tx=sorted({start+i*step for i in range(-3,nt+1)}|{x for x in local_t if start<=x<=horizon}|{w.t-w.ht,w.t})
    kl=min(F(1),w.k-w.hk)/2;kr=F(3,2)*max(F(1),w.k+w.hk)
    kx=sorted({kl+(kr-kl)*j/nk for j in range(nk+1)}|
              {w.k-w.hk+F(j,8)*2*w.hk for j in range(9)}|{F(1)}|{x for x in quote_strikes if kl<x<kr})
    return dict(time_knots=list(map(str,tx)),strike_knots=list(map(str,kx)),
                horizon=str(horizon),start=str(start),nt=nt,nk=nk)


def constraints(window,amin,amax,config,subdivisions=1,progress=QUIET):
    tx=tuple(map(F,config['time_knots']));kx=tuple(map(F,config['strike_knots']))
    horizon=F(config['horizon']);start=F(config.get('start','0'));w=window
    if (subdivisions<1 or len(tx)<8 or len(kx)<5 or tx[-1]!=horizon or
        horizon<=start or start<0 or start>=w.t-w.ht or horizon>w.t+w.ht or tx[0]>=start or start not in tx or
        kx[0]<=0 or kx[0]>=min(F(1),w.k-w.hk) or kx[-1]<=max(F(1),w.k+w.hk) or
        any(a>=b for a,b in zip(tx[:-1],tx[1:])) or any(a>=b for a,b in zip(kx[:-1],kx[1:]))):
        raise ValueError('Invalid denominator subsolution support')
    if any(z not in tx for z in (w.t-w.ht,w.t) if start<z<horizon):
        raise ValueError('Time knots must split the polynomial source')
    if any(z not in kx for z in (w.k-w.hk,w.k,w.k+w.hk)):
        raise ValueError('Strike knots must split the polynomial source')
    nt,nk=len(tx)-4,len(kx)-4;rows=[];rhs=[]
    pt=bump(w.t,w.ht);pk=mul([0,0,1],bump(w.k,w.hk))
    for it,(ta,tb) in enumerate(zip(tx[:-1],tx[1:])):
        if tb<=start:continue
        its=range(max(0,it-3),min(nt-1,it)+1)
        for jk,(ka,kb) in enumerate(zip(kx[:-1],kx[1:])):
            jks=range(max(0,jk-3),min(nk-1,jk)+1)
            for st in range(subdivisions):
                tlo=ta+(tb-ta)*st/subdivisions;thi=ta+(tb-ta)*(st+1)/subdivisions
                bt={i:bernstein(spline_piece(tx,i,3,it),tlo,thi,6) for i in its}
                bdt={i:bernstein(deriv(spline_piece(tx,i,3,it)),tlo,thi,6) for i in its}
                ft=bernstein(pt if ta>=w.t-w.ht else [F(0)],tlo,thi,6)
                for sk in range(subdivisions):
                    klo=ka+(kb-ka)*sk/subdivisions;khi=ka+(kb-ka)*(sk+1)/subdivisions
                    bk={j:bernstein(spline_piece(kx,j,3,jk),klo,khi,8) for j in jks}
                    bg={j:bernstein(mul([0,0,1],deriv(deriv(spline_piece(kx,j,3,jk)))),klo,khi,8) for j in jks}
                    fk=bernstein(pk if ka>=w.k-w.hk and kb<=w.k+w.hk else [F(0)],klo,khi,8)
                    for r in range(7):
                        for s in range(9):
                            source=ft[r]*fk[s]
                            for a in (amin,amax):
                                row={i*nk+j:-(bdt[i][r]*bk[j][s]+a*bt[i][r]*bg[j][s]/2) for i in its for j in jks}
                                rows.append({j:v for j,v in row.items() if v});rhs.append(source)
        progress.event('denominator_subsolution_boxes','running',done=it+1,total=len(tx)-1)
    bt=basis_values(tx,start);bk=basis_values(kx,F(1))
    objective=[a*b for a in bt for b in bk]
    return rows,rhs,objective


def evaluate(rows,rhs,objective,theta,horizon):
    epsilon=max([F(0)]+[sum(v*theta[j] for j,v in row.items())-b for row,b in zip(rows,rhs)])
    initial=sum(v*x for v,x in zip(objective,theta))
    return initial-epsilon*horizon,epsilon,initial


def search(window,amin,amax,nt=12,nk=16,subdivisions=1,time_limit=60,progress=QUIET,solver='simplex'):
    amin,amax=map(rational,(amin,amax))
    if not 0<amin<=amax:raise ValueError('Positive variance bounds required')
    config=definition(window,nt,nk)
    with progress.phase('denominator_subsolution_assembly',nt=nt,nk=nk):
        rows,rhs,f=constraints(window,amin,amax,config,subdivisions,progress)
    rr=[];cc=[];vv=[];b=[]
    for i,(row,right) in enumerate(zip(rows,rhs)):
        scale=max([abs(x) for x in row.values()]+[abs(right),F(1)])
        for j,x in row.items():rr.append(i);cc.append(j);vv.append(float(x/scale))
        rr.append(i);cc.append(len(f));vv.append(float(-1/scale))
        b.append(float(right/scale))
    A=sparse.coo_matrix((vv,(rr,cc)),shape=(len(rows),len(f)+1)).tocsr()
    cost=np.r_[-np.asarray(f,float),float(F(config['horizon']))];cost/=max(np.max(abs(cost)),1e-30)
    res=optimize(cost,A,b,np.r_[np.full(len(f),-1.),0.],np.ones(len(f)+1),time_limit,
                 solver=solver,progress=progress,label='denominator_subsolution_optimization')
    proposal=res.x[:-1] if res.x is not None else np.zeros(len(f))
    theta=[F.from_float(float(x)) for x in proposal]
    with progress.phase('denominator_subsolution_exact'):
        lower,epsilon,initial=evaluate(rows,rhs,f,theta,F(config['horizon']))
    return dict(kind='ito_spline_subsolution_v1',window=window.json(),amin=str(amin),amax=str(amax),
        definition=config,subdivisions=subdivisions,theta=list(map(str,theta)),
        epsilon=str(epsilon),initial_spline_value=str(initial),lower_bound=str(lower),
        dyadic_lower=str(min(F(1,2),dyadic_floor(lower))),positive=lower>0,
        numerical_optimization_converged=res.success,
        numerical_solver_message=res.message,
        approximate=float(lower),proof='Global affine correction absorbs the exact Bernstein defect; Ito at X0=1')


def verify(doc,window=None,amin=None,amax=None,progress=QUIET):
    w=Window.make(**doc['window']);lo,hi=F(doc['amin']),F(doc['amax'])
    if window is not None and w!=window:raise ValueError('Different denominator kernel')
    if amin is not None and (lo!=F(amin) or hi!=F(amax)):raise ValueError('Different denominator class')
    if not 0<lo<=hi:raise ValueError('Invalid denominator class')
    if F(doc['definition'].get('start','0'))!=0:raise ValueError('Initial-state subsolution must start at time zero')
    rows,rhs,f=constraints(w,lo,hi,doc['definition'],doc['subdivisions'],progress)
    theta=list(map(F,doc['theta']))
    if len(theta)!=len(f):raise ValueError('Invalid subsolution coefficient count')
    bound,epsilon,initial=evaluate(rows,rhs,f,theta,F(doc['definition']['horizon']))
    if (str(bound)!=doc['lower_bound'] or doc['approximate']!=float(bound) or str(epsilon)!=doc['epsilon'] or
        str(initial)!=doc['initial_spline_value'] or doc['positive']!=(bound>0) or
        F(doc['dyadic_lower'])!=min(F(1,2),dyadic_floor(bound))):
        raise ValueError('Incorrect global denominator certificate')
    return bound


def static_rows(config,quotes,n):
    """g(x) <= P(start,x)-epsilon*(H-start), globally in x>=0."""
    tx=tuple(map(F,config['time_knots']));kx=tuple(map(F,config['strike_knots']))
    start=F(config['start']);horizon=F(config['horizon']);dt=horizon-start
    bt=basis_values(tx,start);nk=len(kx)-4;m=len(quotes)
    eps,b0,b1=n,n+1,n+2;plus=n+3;minus=plus+m
    rows=[]
    for cell,(ka,kb) in enumerate(zip(kx[:-1],kx[1:])):
        vals={j:bernstein(spline_piece(kx,j,3,cell),ka,kb,3) for j in range(max(0,cell-3),min(nk-1,cell)+1)}
        for r in range(4):
            x=ka+(kb-ka)*r/3;row={eps:dt,b0:F(1),b1:x}
            for i,tv in enumerate(bt):
                for j,v in vals.items():
                    if tv and v[r]:row[i*nk+j]=-tv*v[r]
            for j,q in enumerate(quotes):
                payoff=max(F(0),x-q.k)
                if payoff:row[plus+j]=payoff;row[minus+j]=-payoff
            rows.append(row)
    rows.append({eps:dt,b0:F(1)})  # x=0, all calls have positive strikes
    tail={b1:F(1)}
    for j in range(m):tail[plus+j]=F(1);tail[minus+j]=F(-1)
    rows.append(tail)
    return rows


def static_evaluate(rows,rhs,config,quotes,theta,b0,b1,lam):
    horizon=F(config['horizon']);start=F(config['start']);n=len(theta)
    epsilon=max([F(0)]+[sum(v*theta[j] for j,v in row.items())-b for row,b in zip(rows,rhs)])
    b1-=max(F(0),b1+sum(lam))
    positive=[max(F(0),x) for x in lam];negative=[max(F(0),-x) for x in lam]
    vector=theta+[epsilon,b0,b1]+positive+negative
    eta=max([F(0)]+[sum(v*vector[j] for j,v in row.items()) for row in static_rows(config,quotes,n)[:-1]])
    b0-=eta
    bound=b0+b1+sum(x*(q.bid if x>=0 else q.ask) for x,q in zip(lam,quotes))
    return bound,epsilon,eta,b0,b1


def search_static(quotes,window,amin,amax,nt=8,nk=10,subdivisions=1,time_limit=60,progress=QUIET):
    amin,amax=map(rational,(amin,amax))
    dates=[q.t for q in quotes if q.t<window.t-window.ht]
    if not dates:return None
    start=max(dates);same=[q for q in quotes if q.t==start]
    config=definition(window,nt,nk,start,[q.k for q in same])
    kx=list(map(F,config['strike_knots']));used=[q for q in same if kx[0]<q.k<kx[-1]]
    if not used:return None
    with progress.phase('denominator_static_assembly',nt=nt,nk=nk):
        rows,rhs,f=constraints(window,amin,amax,config,subdivisions,progress)
        n=len(f);m=len(used);stat=static_rows(config,used,n)
    rr=[];cc=[];vv=[];b=[]
    for i,(row,right) in enumerate(zip(rows+stat,rhs+[F(0)]*len(stat))):
        row=dict(row)
        if i<len(rows):row[n]=F(-1)
        scale=max([abs(x) for x in row.values()]+[abs(right),F(1)])
        for j,x in row.items():rr.append(i);cc.append(j);vv.append(float(x/scale))
        b.append(float(right/scale))
    count=n+3+2*m;A=sparse.coo_matrix((vv,(rr,cc)),shape=(len(b),count)).tocsr()
    cost=np.r_[np.zeros(n+1),-1.,-1.,[-float(q.bid) for q in used],[float(q.ask) for q in used]]
    lower=np.r_[np.full(n,-1.),0.,-20.,-20.,np.zeros(2*m)]
    upper=np.r_[np.ones(n+1),20.,20.,np.full(2*m,20.)]
    res=optimize(cost,A,b,lower,upper,time_limit,solver='simplex',progress=progress,label='denominator_static_optimization')
    x=list(map(lambda v:F.from_float(float(v)),res.x if res.x is not None else np.zeros(count)))
    theta=x[:n];lam=[x[n+3+j]-x[n+3+m+j] for j in range(m)]
    with progress.phase('denominator_static_exact'):
        bound,epsilon,eta,b0,b1=static_evaluate(rows,rhs,config,used,theta,x[n+1],x[n+2],lam)
    return dict(kind='ito_static_subsolution_v1',window=window.json(),amin=str(amin),amax=str(amax),
        definition=config,subdivisions=subdivisions,theta=list(map(str,theta)),
        epsilon=str(epsilon),portfolio=dict(cash=str(b0),stock=str(b1),weights=list(map(str,lam))),
        quotes=quote_payload(used),lower_bound=str(bound),dyadic_lower=str(min(F(1,2),dyadic_floor(bound))),
        positive=bound>0,approximate=float(bound),numerical_optimization_converged=res.success,
        numerical_solver_message=res.message,
        proof='Ito subsolution on [quote maturity,H], exact defect correction, global static subhedge at observed bid/ask')


def verify_static(doc,quotes,window,amin,amax,progress=QUIET):
    if not 0<F(amin)<=F(amax):raise ValueError('Invalid static denominator class')
    if doc['window']!=window.json() or F(doc['amin'])!=F(amin) or F(doc['amax'])!=F(amax):
        raise ValueError('Different static denominator problem')
    used=payload_quotes(doc['quotes']);observed={q.id:q for q in quotes}
    start=F(doc['definition']['start'])
    if not used or any(q.id not in observed or q!=observed[q.id] or q.t!=start for q in used):
        raise ValueError('Static portfolio not linked to observations')
    kx=list(map(F,doc['definition']['strike_knots']))
    if any(not kx[0]<q.k<kx[-1] or q.k not in kx for q in used):raise ValueError('Invalid static strike partition')
    rows,rhs,f=constraints(window,F(amin),F(amax),doc['definition'],doc['subdivisions'],progress)
    theta=list(map(F,doc['theta']));lam=list(map(F,doc['portfolio']['weights']))
    if len(theta)!=len(f) or len(lam)!=len(used):raise ValueError('Incorrect static certificate dimensions')
    b0,b1=F(doc['portfolio']['cash']),F(doc['portfolio']['stock'])
    bound,epsilon,eta,b0check,b1check=static_evaluate(rows,rhs,doc['definition'],used,theta,b0,b1,lam)
    if eta or b0check!=b0 or b1check!=b1 or str(epsilon)!=doc['epsilon'] or str(bound)!=doc['lower_bound'] or doc['approximate']!=float(bound) or (
        doc['positive']!=(bound>0) or F(doc['dyadic_lower'])!=min(F(1,2),dyadic_floor(bound))):
        raise ValueError('Incorrect static denominator certificate')
    return F(doc['dyadic_lower'])


def build(quotes,window,amin,amax,levels=(8,),time_limit=60,progress=QUIET):
    brownian=brownian_floor(window,amin,amax);best=F(brownian['dyadic_lower']);records=[]
    progress.event('denominator_brownian','certified',lower=float(best))
    for cells in levels:
        doc=search_static(quotes,window,amin,amax,cells,cells+2,time_limit=time_limit,progress=progress)
        if doc is not None:
            records.append(doc);best=max(best,F(doc['dyadic_lower']))
            progress.event('denominator_global_floor','certified',lower=float(best),cells=cells)
    return dict(kind='global_denominator_bundle_v1',window=window.json(),amin=str(amin),amax=str(amax),
        brownian=brownian,static_subsolutions=records,lower_bound=str(best),approximate=float(best),positive=best>0)


def verify_bundle(doc,quotes,window,amin,amax,progress=QUIET):
    if doc['window']!=window.json() or F(doc['amin'])!=F(amin) or F(doc['amax'])!=F(amax):
        raise ValueError('Different denominator bundle problem')
    best=verify_brownian(doc['brownian'],window,amin,amax)
    for record in doc['static_subsolutions']:best=max(best,verify_static(record,quotes,window,amin,amax,progress))
    if best<=0 or str(best)!=doc['lower_bound'] or doc['approximate']!=float(best) or not doc['positive']:raise ValueError('Incorrect positive denominator floor')
    return best
