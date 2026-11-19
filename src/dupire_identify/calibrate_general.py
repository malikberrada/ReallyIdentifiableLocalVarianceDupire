"""Direct local-variance proposals for arbitrary finite European quote sets.

No Black-mixture baseline is required. FD prices, Jacobians and objective
values are proposals only; fixed_model verifies the continuous diffusion.
"""
from fractions import Fraction as F
from pathlib import Path
import json
import time
import math
import numpy as np
from scipy.special import ndtr
from scipy.linalg import solve_banded
from scipy.optimize import least_squares,minimize,brentq
from . import plateau
from .io import save_json
from .progress import QUIET

try:
    from ._native import fd_batch as _native_batch
except ImportError:
    _native_batch=None


class BudgetStop(RuntimeError):pass


def black(k,v):
    k=np.asarray(k);v=np.asarray(v)
    s=np.sqrt(v);d=(-np.log(k)+v/2)/s
    return ndtr(d)-k*ndtr(d-s)


def template(quotes,family,amin,amax,controls=6):
    amin,amax=F(amin),F(amax);w=family.center();ta,tb,ka,kb=family.support()
    prefix=min(min(q.t for q in quotes),ta)/2
    horizon=max(max(q.t for q in quotes),tb)
    if horizon*amax>30:raise ValueError('Proposal horizon exceeds explicit numerical budget; use a supplied verified model')
    left=min(F(1,16),ka/2,min(q.k for q in quotes)/2)
    right=max(F(8),kb*2,max(q.k for q in quotes)*2)
    innerlo=min(min(q.k for q in quotes),w.k-w.hk)*F(9,10)
    innerhi=max(max(q.k for q in quotes),w.k+w.hk)*F(11,10)
    if controls<2:raise ValueError('At least two interior controls required')
    ks=sorted({F(round(float(x)*4096),4096) for x in np.geomspace(float(innerlo),float(innerhi),controls)}|{w.k-w.hk,w.k+w.hk})
    knots=[left]+ks+[right]
    times=sorted({prefix,horizon,w.t-w.ht,w.t+w.ht}|{q.t for q in quotes})
    # Infer just an initial scale; its entire strike-dependent evolution is free.
    implied=[]
    for q in quotes:
        mid=float((q.bid+q.ask)/2)
        l,h=float(amin),float(amax)
        if black(float(q.k),l*float(q.t))<=mid<=black(float(q.k),h*float(q.t)):
            implied.append(brentq(lambda a:float(black(float(q.k),a*float(q.t)))-mid,l,h))
    a0=float(np.median(implied)) if implied else float((amin+amax)/2)
    a0=min(float(amax),max(float(amin),a0))
    config=dict(kind='piecewise_quintic_plateau_diffusion_v1',amin=str(amin),amax=str(amax),
        prefix_end=str(prefix),prefix_variance=str(F(a0)),strike_knots=list(map(str,knots)),
        price_envelope_intersection=True,
        phases=[dict(left=str(a),right=str(b),variance=[str(F(a0))]*len(knots)) for a,b in zip(times,times[1:])])
    return config


def fd_batch_python(grid,coefficients,lefts,rights,initial,steps,window):
    """Independent M-matrix FD proposal, SDIRK2, native LAPACK fallback."""
    Bn,P,N=coefficients.shape;curves=np.empty((Bn,P,N));nums=np.zeros(Bn);dens=np.zeros(Bn)
    h=np.diff(grid);ql=2/(h[:-1]*(h[:-1]+h[1:]));qr=2/(h[1:]*(h[:-1]+h[1:]));qd=-ql-qr
    weights=np.r_[h[0]/2,(h[:-1]+h[1:])/2,h[-1]/2]
    T,K,ht,hk=window;pk=np.maximum(0,1-((grid-K)/hk)**2)**3
    gamma=1-1/np.sqrt(2.)
    for model in range(Bn):
        c=initial[model].copy()
        for p,(a,b) in enumerate(zip(lefts,rights)):
            count=max(1,int(np.ceil((b-a)*steps)));dt=(b-a)/count
            D=.5*coefficients[model,p,1:-1]*grid[1:-1]**2
            L,Dg,U=D*ql,D*qd,D*qr
            A=np.zeros((3,N-2));A[0,1:]=-gamma*dt*U[:-1];A[1]=1-gamma*dt*Dg;A[2,:-1]=-gamma*dt*L[1:]
            for i in range(count):
                old=c.copy();rhs=c[1:-1].copy();rhs[0]+=gamma*dt*L[0]*c[0];rhs[-1]+=gamma*dt*U[-1]*c[-1]
                y=np.r_[c[0],solve_banded((1,1),A,rhs),c[-1]]
                rhs=c[1:-1]+(1-gamma)*dt*(L*y[:-2]+Dg*y[1:-1]+U*y[2:])
                rhs[0]+=gamma*dt*L[0]*c[0];rhs[-1]+=gamma*dt*U[-1]*c[-1]
                c=np.r_[c[0],solve_banded((1,1),A,rhs),c[-1]]
                mid=a+(i+.5)*dt;pt=max(0.,1-((mid-T)/ht)**2)**3
                if pt:
                    av=(old+c)/2;density=np.r_[0.,np.maximum(0,ql*av[:-2]+qd*av[1:-1]+qr*av[2:]),0.]
                    mass=pt*pk*grid*grid*density*weights
                    dens[model]+=dt*np.sum(mass);nums[model]+=dt*np.sum(mass*coefficients[model,p])
            curves[model,p]=c
    return curves,nums,dens


class Proposal:
    def __init__(self,config,quotes,window,nodes=320,steps=512,threads=1,max_evaluations=2000,max_seconds=180,progress=QUIET,backend='auto'):
        if nodes<20 or steps<1 or threads<1 or max_evaluations<1 or max_seconds<=0:raise ValueError('Positive proposal budgets required')
        plateau.validate_config(config);self.config=config;self.quotes=quotes;self.window=window;self.progress=progress
        self.knots=np.array(list(map(float,map(F,config['strike_knots']))))
        if len(self.knots)<4:raise ValueError('At least two interior controls required')
        self.ninside=len(self.knots)-2;self.P=len(config['phases']);self.full_n=1+self.P*self.ninside
        self.full_initial=np.r_[float(F(config['prefix_variance'])),[float(F(v)) for p in config['phases'] for v in p['variance'][1:-1]]]
        locked=set()
        if config.get('target_plateau'):
            ta,tb,ka,kb=map(F,config['target_plateau']['support'])
            for p,phase in enumerate(config['phases']):
                if ta<=F(phase['left']) and F(phase['right'])<=tb:
                    locked.update(1+p*self.ninside+j for j,k in enumerate(map(F,config['strike_knots'][1:-1])) if ka<=k<=kb)
        self.free=np.array([i for i in range(self.full_n) if i not in locked],dtype=int);self.n=len(self.free)
        self.grid=np.unique(np.r_[np.geomspace(self.knots[0],self.knots[-1],nodes),self.knots,[float(q.k) for q in quotes]])
        self.lefts=np.array([float(F(p['left'])) for p in config['phases']]);self.rights=np.array([float(F(p['right'])) for p in config['phases']])
        j=np.clip(np.searchsorted(self.knots,self.grid,side='right')-1,0,len(self.knots)-2)
        u=np.clip((self.grid-self.knots[j])/(self.knots[j+1]-self.knots[j]),0,1);s=u**3*(10-15*u+6*u*u)
        basis=np.zeros((len(self.grid),len(self.knots)))
        basis[np.arange(len(self.grid)),j]=1-s;basis[np.arange(len(self.grid)),j+1]+=s
        self.basis=basis[:,1:-1].copy();self.basis[:,0]+=basis[:,0];self.basis[:,-1]+=basis[:,-1]
        self.qp=[next(i for i,p in enumerate(config['phases']) if F(p['right'])==q.t) for q in quotes]
        self.qk=[int(np.searchsorted(self.grid,float(q.k))) for q in quotes]
        self.low=float(F(config['amin']));self.high=float(F(config['amax']));self.steps=steps;self.threads=threads
        self.bid=np.array([float(q.bid) for q in quotes]);self.ask=np.array([float(q.ask) for q in quotes]);self.width=np.maximum(self.ask-self.bid,1e-10)
        intrinsic=np.array([max(0.,1-float(q.k)) for q in quotes])
        self.margin_high=np.minimum(self.width/6,1e-3)
        self.margin_low=np.where(self.bid>intrinsic+1e-12,self.margin_high,0.)
        self.started=time.monotonic();self.max_seconds=max_seconds;self.max_evaluations=max_evaluations;self.evaluations=0
        self.best=None;self.best_error=np.inf;self.cache=None
        self.endpoint_mode=None;self.endpoint_best=None;self.endpoint_value=None
        from .fd_acceleration import BatchEngine
        self.engine=BatchEngine(backend,threads,progress)
        self.started=time.monotonic()

    def initial(self):return self.full_initial[self.free].copy()

    def expand(self,thetas):
        z=np.atleast_2d(thetas)
        full=np.broadcast_to(self.full_initial,(len(z),self.full_n)).copy();full[:,self.free]=z
        return full

    def evaluate(self,thetas):
        z=np.atleast_2d(thetas);count=len(z)
        if self.evaluations+count>self.max_evaluations or time.monotonic()-self.started>self.max_seconds:raise BudgetStop('Explicit proposal budget reached')
        full=self.expand(z)
        coeff=np.einsum('bpc,nc->bpn',full[:,1:].reshape(count,self.P,self.ninside),self.basis)
        init=black(self.grid[None,:],full[:,0,None]*float(F(self.config['prefix_end'])))
        init[:,0]=1-self.grid[0];init[:,-1]=0.
        w=np.array([float(x) for x in (self.window.t,self.window.k,self.window.ht,self.window.hk)])
        curves,ns,ds=self.engine(self.grid,coeff,self.lefts,self.rights,init,self.steps,w)
        prices=curves[:,self.qp,self.qk];averages=np.divide(ns,ds,out=np.full(count,(self.low+self.high)/2),where=np.asarray(ds)>0)
        if np.any(~np.isfinite(prices)) or np.any(~np.isfinite(averages)):raise RuntimeError('Nonfinite FD proposal')
        self.evaluations+=count
        errors=np.max(np.maximum(self.bid+self.margin_low-prices,prices-self.ask+self.margin_high)/self.width,axis=1)
        for i,e in enumerate(errors):
            if e<self.best_error:self.best_error=float(e);self.best=z[i].copy()
            if e<=0 and self.endpoint_mode:
                v=averages[i]
                if self.endpoint_value is None or (v<self.endpoint_value if self.endpoint_mode=='minimum' else v>self.endpoint_value):
                    self.endpoint_best=z[i].copy();self.endpoint_value=float(v)
        self.progress.event('general_diffusion_proposals','running',done=self.evaluations,total=self.max_evaluations,
            elapsed_seconds=round(time.monotonic()-self.started,2),best_spread_violation=self.best_error,
            engine=self.engine.selected)
        return prices,averages

    def residual(self,theta):
        p,a=self.evaluate(theta);p=p[0]
        # A small margin helps the later continuous-price enclosure. Intrinsic
        # and nonnegative comparisons may legitimately touch a zero bid.
        low=self.bid+self.margin_low;high=self.ask-self.margin_high
        r=(p-np.clip(p,low,high))/self.width
        return np.r_[r,1e-5*(theta-self.initial())/(self.high-self.low)]

    def jacobian(self,theta):
        step=np.maximum(1e-6*(self.high-self.low),abs(theta)*1e-4)
        step=np.where(theta+step<=self.high,step,-step)
        batch=np.tile(theta,(self.n+1,1));batch[1:]+=np.diag(step)
        p,a=self.evaluate(batch)
        r=(p-np.clip(p,self.bid+self.margin_low,self.ask-self.margin_high))/self.width
        return np.vstack(((r[1:]-r[0]).T/step,np.eye(self.n)*1e-5/(self.high-self.low)))

    def to_config(self,theta):
        out=json.loads(json.dumps(self.config));theta=self.expand(np.clip(theta,self.low,self.high))[0]
        def exact(x):
            if abs(x-self.low)<1e-12:return F(self.config['amin'])
            if abs(x-self.high)<1e-12:return F(self.config['amax'])
            return F(float(x))
        out['prefix_variance']=str(exact(theta[0]))
        controls=theta[1:].reshape(self.P,self.ninside)
        for p,c in zip(out['phases'],controls):p['variance']=list(map(lambda x:str(exact(x)),np.r_[c[0],c,c[-1]]))
        return out

    def fit(self,endpoint=None):
        self.endpoint_mode=endpoint
        x=self.initial()
        try:
            result=least_squares(self.residual,x,jac=self.jacobian,bounds=(self.low,self.high),
                max_nfev=max(5,self.max_evaluations//max(1,self.n+1)),ftol=1e-7,xtol=1e-7,gtol=1e-7)
            x=result.x
        except BudgetStop:pass
        if self.best is not None and self.best_error<0:x=self.best.copy()
        if endpoint and self.evaluations+self.n+2<=self.max_evaluations:
            sign=1 if endpoint=='minimum' else -1
            cache={}
            def get(z):
                key=np.asarray(z).tobytes()
                if key not in cache:
                    h=np.maximum(1e-6*(self.high-self.low),abs(z)*1e-4);h=np.where(z+h<=self.high,h,-h)
                    batch=np.tile(z,(self.n+1,1));batch[1:]+=np.diag(h)
                    prices,av=self.evaluate(batch)
                    cache.clear();cache[key]=(prices[0],av[0],(prices[1:]-prices[0]).T/h,(av[1:]-av[0])/h)
                return cache[key]
            def fun(z):return sign*get(z)[1]
            def jac(z):return sign*get(z)[3]
            def con(z):
                p=get(z)[0];return np.r_[(p-self.bid-self.margin_low)/self.width,(self.ask-p-self.margin_high)/self.width]
            def cjac(z):
                j=get(z)[2]/self.width[:,None];return np.vstack([j,-j])
            try:
                result=minimize(fun,x,jac=jac,bounds=[(self.low,self.high)]*self.n,
                    constraints=[dict(type='ineq',fun=con,jac=cjac)],method='SLSQP',options=dict(maxiter=40,ftol=1e-7))
                if result.x is not None:x=result.x
            except BudgetStop:pass
        if endpoint and self.endpoint_best is not None:x=self.endpoint_best.copy()
        return self.to_config(x),dict(numerical_spread_violation=self.best_error,evaluated_trajectories=self.evaluations,
            elapsed_s=time.monotonic()-self.started,certified=False,
            batch_backend=self.engine.selected,batch_timing=self.engine.timing,endpoint_value_proposal=self.endpoint_value,
            note='FD proposals require an independent continuous PDE certificate')


def target_template(quotes,family,amin,amax,endpoint,controls=6,seed=None):
    """Lock an entire family support at an extreme; fit compensating controls.

    This changes only the proposal family. The independent coefficient and
    continuous-price verifiers are still the authority for feasibility.
    """
    from .fixed_model import variance_at
    cfg=template(quotes,family,amin,amax,controls)
    ta,tb,ka,kb=family.support();target=F(amin if endpoint=='minimum' else amax)
    old=list(map(F,cfg['strike_knots']));left,right=old[0],old[-1]
    ramp=min((kb-ka)/4,ka/4,F(1,20))
    anchors=sorted({left,max(left,ka-ramp),ka,kb,min(right,kb+ramp),right})
    # Nearly coincident geometric controls produce tiny coefficient panels,
    # then expensive graded proof meshes. Preserve exact plateau anchors and
    # keep other controls only when they have meaningful room on both sides.
    knots=list(anchors)
    for k in old[1:-1]:
        if all(abs(k-z)>=ramp/2 for z in knots):knots.append(k)
    knots.sort()
    prefix=F(cfg['prefix_end']);end=F(cfg['phases'][-1]['right'])
    dates={prefix,end,ta,tb}|{q.t for q in quotes}
    if seed is not None:
        dates|={F(seed['prefix_end'])}|{F(p[k]) for p in seed['phases'] for k in ('left','right')}
        cfg['prefix_variance']=seed['prefix_variance']
    dates=sorted(t for t in dates if prefix<=t<=end)
    phases=[]
    for a,b in zip(dates,dates[1:]):
        values=[];mid=(a+b)/2
        for k in knots:
            v=F(cfg['prefix_variance'])
            if seed is not None:
                for p,z in enumerate(seed['phases']):
                    if F(z['left'])<=mid<=F(z['right']):v=variance_at(seed,p,k);break
            if ta<=a and b<=tb and ka<=k<=kb:v=target
            values.append(str(v))
        phases.append(dict(left=str(a),right=str(b),variance=values))
    cfg.update(strike_knots=list(map(str,knots)),phases=phases,
        target_plateau=dict(endpoint=endpoint,support=list(map(str,(ta,tb,ka,kb))),target=str(target)))
    plateau.validate_config(cfg)
    return cfg
