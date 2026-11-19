"""Continuous diffusion witnesses: analytic Black mixture + compact C² splines.

Both inequalities amin*k²c_kk <= 2*c_T <= amax*k²c_kk and a strict density
floor are verified everywhere on the correction support. Outside, the analytic
baseline is unchanged. Pricing the model does not require a numerical PDE.
"""
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import time
from copy import deepcopy
from scipy import sparse
from scipy.optimize import linprog
from .interval import I,black_call
from .polynomial import deriv,mul,bump,integrate
from .curves import spline_piece,bernstein,value
from .density import gamma_box,density_integral
from .io import rational,save_json,quote_payload
from .progress import QUIET
from .native_solver import optimize,WarmLP


def baseline(quotes,variances,amin,amax,progress=QUIET):
    variances=list(map(rational,variances))
    if not variances or any(not amin<a<amax for a in variances):
        raise ValueError('Baseline variances must be strictly inside (amin,amax)')
    prices=[]
    for qi,q in enumerate(quotes):
        prices.append([black_call(q.k,a*q.t) for a in variances])
        progress.event('witness_baseline_prices','running',done=qi+1,total=len(quotes))
    if len(variances)==1:weights=[F(1)]
    else:
        A=[];b=[]
        for quote,row in zip(quotes,prices):
            width=max(float(quote.ask-quote.bid),1e-12)
            A.append([float(p.hi) for p in row]+[width]);b.append(float(quote.ask))
            A.append([-float(p.lo) for p in row]+[width]);b.append(-float(quote.bid))
        with progress.phase('witness_baseline_calibration',atoms=len(variances)):
            res=linprog([0]*len(variances)+[-1],A_ub=A,b_ub=b,
                A_eq=[[1]*len(variances)+[0]],b_eq=[1],bounds=[(0,1)]*len(variances)+[(0,None)],method='highs')
        if not res.success:raise ValueError('No compatible analytic mixture baseline in the proposed atoms')
        weights=[max(F(0),F.from_float(float(x))) for x in res.x[:-1]]
        total=sum(weights);weights=[x/total for x in weights]
    intervals=[]
    for q,row in zip(quotes,prices):
        p=sum((w*c for w,c in zip(weights,row)),I(0)).rounded()
        if p.lo<q.bid or p.hi>q.ask:
            raise ValueError('No certified compatible baseline; propose other baseline variances or a valid baseline')
        intervals.append(p)
    return dict(variances=list(map(str,variances)),weights=list(map(str,weights)),
                quote_prices=[p.json() for p in intervals]),intervals


def spec(window,quotes,nt=10,nk=12,time_patch=None,strike_patch=None):
    w=window
    if nt<4 or nk<4:raise ValueError('At least four spline cells per dimension required')
    if time_patch is None:
        left=max([F(0)]+[q.t for q in quotes if q.t<=w.t-w.ht])
        right=min([q.t for q in quotes if q.t>=w.t+w.ht] or [w.t+2*w.ht])
        if left==0:left=(w.t-w.ht)/2
    else:left,right=map(rational,time_patch)
    if strike_patch is None:
        kl=max(F(1,100),w.k-4*w.hk);kr=w.k+4*w.hk
    else:kl,kr=map(rational,strike_patch)
    if not 0<left<=w.t-w.ht<w.t+w.ht<=right or not 0<kl<=w.k-w.hk<w.k+w.hk<=kr:
        raise ValueError('Spline patch must contain the full target kernel, with positive coordinates')
    return dict(time_knots=[str(left+(right-left)*i/nt) for i in range(nt+1)],
                strike_knots=[str(kl+(kr-kl)*j/nk) for j in range(nk+1)])


def expanded_spec(window,quotes,target_cells=8,reference=None,expand_dates=False):
    """Nested local refinement; free the marginal laws at observed dates."""
    if target_cells<4:raise ValueError('At least four target cells required')
    old=reference or spec(window,quotes,12,12)
    ts=set(map(F,old['time_knots']));ks=set(map(F,old['strike_knots']))
    if expand_dates:
        left=min(min(ts),min(min(q.t for q in quotes),window.t-window.ht)/2)
        right=max(max(ts),max(max(q.t for q in quotes),window.t+window.ht)+window.ht)
        a,b=min(ts),max(ts)
        ts.update(left+(a-left)*i/4 for i in range(5))
        ts.update(b+(right-b)*i/4 for i in range(5))
        ts.update(q.t for q in quotes if left<q.t<right)
    ts.update(window.t-window.ht+2*window.ht*i/target_cells for i in range(target_cells+1))
    ks.update(window.k-window.hk+2*window.hk*i/target_cells for i in range(target_cells+1))
    return dict(time_knots=list(map(str,sorted(ts))),strike_knots=list(map(str,sorted(ks))))


def refine_curve(old_knots,coefficients,new_knots):
    """Exact Boehm knot insertion, including zero extension of compact splines."""
    knots=list(map(F,old_knots));c=list(map(F,coefficients));new=list(map(F,new_knots))
    if len(c)!=len(knots)-4 or not set(knots)<=set(new):raise ValueError('Refinement must contain every old knot')
    if any(a>=b for a,b in zip(knots[:-1],knots[1:])) or any(a>=b for a,b in zip(new[:-1],new[1:])):
        raise ValueError('Knot vectors must be strictly increasing')
    for knot in [x for x in new if x<knots[0]][::-1]:knots.insert(0,knot);c.insert(0,F(0))
    for knot in [x for x in new if x>knots[-1]]:knots.append(knot);c.append(F(0))
    for knot in new:
        if knot in knots:continue
        span=next(i for i in range(len(knots)-1) if knots[i]<knot<knots[i+1])
        out=[]
        for i in range(len(c)+1):
            alpha=F(1) if i<=span-3 else (F(0) if i>=span+1 else (knot-knots[i])/(knots[i+3]-knots[i]))
            out.append(alpha*(c[i] if i<len(c) else 0)+(1-alpha)*(c[i-1] if i else 0))
        knots.insert(span+1,knot);c=out
    return c


def refine_theta(doc,definition):
    old=doc['definition'];ot,ok=old['time_knots'],old['strike_knots']
    nt,nk=len(ot)-4,len(ok)-4;theta=list(map(F,doc['theta']))
    newt,newk=definition['time_knots'],definition['strike_knots']
    columns=[refine_curve(ot,[theta[i*nk+j] for i in range(nt)],newt) for j in range(nk)]
    rows=[refine_curve(ok,[columns[j][i] for j in range(nk)],newk) for i in range(len(newt)-4)]
    return [v for row in rows for v in row]


def basis_values(knots,x,order=0):
    n=len(knots)-4
    if x<=knots[0] or x>=knots[-1]:return [F(0)]*n
    cell=next(i for i in range(len(knots)-1) if knots[i]<=x<knots[i+1])
    out=[]
    for i in range(n):
        p=spline_piece(tuple(knots),i,3,cell)
        for _ in range(order):p=deriv(p)
        out.append(value(p,x))
    return out


def constraints(quotes,base,window,definition,amin,amax,subdivisions=1,
                density_floor=F(1,100),theta_bound=F(1),progress=QUIET):
    tx=tuple(map(F,definition['time_knots']));kx=tuple(map(F,definition['strike_knots']))
    if (len(tx)<5 or len(kx)<5 or tx[0]<=0 or kx[0]<=0 or
        any(a>=b for a,b in zip(tx[:-1],tx[1:])) or any(a>=b for a,b in zip(kx[:-1],kx[1:]))):
        raise ValueError('Invalid compact cubic spline support')
    nt,nk=len(tx)-4,len(kx)-4;n=nt*nk
    av=list(map(F,base['variances']));weights=list(map(F,base['weights']))
    if (not av or len(av)!=len(weights) or sum(weights)!=1 or any(x<0 for x in weights) or
        any(not amin<a<amax for a in av)):
        raise ValueError('Invalid certified analytic baseline')
    if not 0<density_floor<1 or theta_bound<=0 or subdivisions<1:raise ValueError('Invalid witness settings')
    rows=[];rhs=[]
    def row(coeff,b):
        coeff={j:v for j,v in coeff.items() if v}
        if b<0:raise ValueError('Baseline is not a feasible contraction anchor')
        if coeff:rows.append(coeff);rhs.append(F(b))
    for i in range(nt*nk):row({i:1},theta_bound);row({i:-1},theta_bound)
    for it,(ta,tb) in enumerate(zip(tx[:-1],tx[1:])):
        its=list(range(max(0,it-3),min(nt-1,it)+1))
        for jk,(ka,kb) in enumerate(zip(kx[:-1],kx[1:])):
            jks=list(range(max(0,jk-3),min(nk-1,jk)+1))
            for st in range(subdivisions):
                tlo,thi=ta+(tb-ta)*st/subdivisions,ta+(tb-ta)*(st+1)/subdivisions
                bt={i:bernstein(spline_piece(tx,i,3,it),tlo,thi,3) for i in its}
                bdt={i:bernstein(deriv(spline_piece(tx,i,3,it)),tlo,thi,3) for i in its}
                for sk in range(subdivisions):
                    klo,khi=ka+(kb-ka)*sk/subdivisions,ka+(kb-ka)*(sk+1)/subdivisions
                    bk={j:bernstein(spline_piece(kx,j,3,jk),klo,khi,3) for j in jks}
                    bg={j:bernstein(mul([0,0,1],deriv(deriv(spline_piece(kx,j,3,jk)))),klo,khi,3) for j in jks}
                    gl=[gamma_box(tlo,thi,klo,khi,a)[0].lo for a in av]
                    density=sum(w*g for w,g in zip(weights,gl))
                    lower=sum(w*(a-amin)*g for w,a,g in zip(weights,av,gl))
                    upper=sum(w*(amax-a)*g for w,a,g in zip(weights,av,gl))
                    for r in range(4):
                        for s in range(4):
                            g={};lo={};hi={}
                            for i in its:
                                for j in jks:
                                    index=i*nk+j
                                    gamma=bt[i][r]*bg[j][s]
                                    time=2*bdt[i][r]*bk[j][s]
                                    g[index]=-gamma
                                    lo[index]=-(time-amin*gamma)
                                    hi[index]=-(amax*gamma-time)
                            row(g,(1-density_floor)*density)
                            row(lo,lower);row(hi,upper)
        progress.event('witness_bernstein_boxes','running',done=it+1,total=len(tx)-1,
                       subdivisions=subdivisions)
    # Quote containment includes all observations, also if patch crosses dates.
    quote_values=[]
    for q in quotes:
        p=sum((w*black_call(q.k,a*q.t) for w,a in zip(weights,av)),I(0)).rounded()
        tv=basis_values(tx,q.t);kv=basis_values(kx,q.k)
        coeff={i*nk+j:u*v for i,u in enumerate(tv) for j,v in enumerate(kv) if u and v}
        row(coeff,q.ask-p.hi);row({j:-v for j,v in coeff.items()},p.lo-q.bid)
        quote_values.append((p,coeff))
    numerator,denominator=functional_coefficients(window,definition)
    return rows,rhs,numerator,denominator,quote_values


def functional_coefficients(window,definition):
    # Exact functional coefficients, via separability of the polynomial kernel.
    tx=tuple(map(F,definition['time_knots']));kx=tuple(map(F,definition['strike_knots']))
    nt,nk=len(tx)-4,len(kx)-4
    pt,pk=bump(window.t,window.ht),bump(window.k,window.hk)
    def integrals(knots,count,phi,low,high,order=0,k2=False):
        out=[]
        for index in range(count):
            total=F(0)
            for cell,(a,b) in enumerate(zip(knots[:-1],knots[1:])):
                l,h=max(a,low),min(b,high)
                if l>=h:continue
                p=list(spline_piece(knots,index,3,cell))
                for _ in range(order):p=deriv(p)
                if k2:p=mul([0,0,1],p)
                total+=integrate(mul(phi,p),l,h)
            out.append(total)
        return out
    ti=integrals(tx,nt,pt,window.t-window.ht,window.t+window.ht)
    td=integrals(tx,nt,pt,window.t-window.ht,window.t+window.ht,1)
    ki=integrals(kx,nk,pk,window.k-window.hk,window.k+window.hk)
    kd=integrals(kx,nk,pk,window.k-window.hk,window.k+window.hk,2,True)
    numerator=[2*a*b for a in td for b in ki]
    denominator=[a*b for a in ti for b in kd]
    return numerator,denominator


def repair(theta,rows,rhs):
    theta=[F.from_float(float(x)) for x in theta]
    scale=F(1)
    for row,b in zip(rows,rhs):
        dot=sum(v*theta[j] for j,v in row.items())
        if dot>b and dot>0:scale=min(scale,b/dot)
    scale*=1-F(1,2**42)
    theta=[scale*x for x in theta]
    slack=min([F(0)]+[b-sum(v*theta[j] for j,v in row.items()) for row,b in zip(rows,rhs)])
    if slack<0:raise ArithmeticError('Exact primal repair failed')
    return theta,scale


def average(theta,n,d,density,variances,weights,density_floor):
    db=sum((w*x for w,x in zip(weights,density)),I(0)).rounded()
    dn=sum(v*x for v,x in zip(d,theta));nn=sum(v*x for v,x in zip(n,theta))
    denom=(db+dn).intersect(I(density_floor*db.lo,db.hi+dn))
    if len(variances)==1:
        answer=variances[0]+(nn-variances[0]*dn)/denom
    else:
        nb=sum((w*a*x for w,a,x in zip(weights,variances,density)),I(0)).rounded()
        answer=(nb+nn)/denom
    return answer.rounded(),denom


def search(quotes,window,amin,amax,base,definition,subdivisions=1,quadrature_cells=16,
           density_floor=F(1,100),theta_bound=F(1),time_limit=120,progress=QUIET,
           endpoints=('minimum','maximum'),initial=None,max_iterations=80,full_threshold=200):
    amin,amax,density_floor,theta_bound=map(rational,(amin,amax,density_floor,theta_bound))
    av=list(map(F,base['variances']));weights=list(map(F,base['weights']))
    with progress.phase('witness_global_constraints'):
        rows,rhs,n,d,quote_values=constraints(quotes,base,window,definition,amin,amax,
            subdivisions,density_floor,theta_bound,progress)
    with progress.phase('witness_certified_quadrature',cells=quadrature_cells):
        density=[density_integral(window,a,quadrature_cells,progress) for a in av]
    db=sum((w*x for w,x in zip(weights,density)),I(0)).rounded()
    nb=sum((w*a*x for w,a,x in zip(weights,av,density)),I(0)).rounded()
    dmid=(db.lo+db.hi)/2;nmid=(nb.lo+nb.hi)/2
    min_approx=density_floor*db.lo-(db.hi-db.lo)/2
    if min_approx<=0:raise ValueError('Quadrature too wide for witness optimization; increase --quadrature-cells')
    nvar=len(n)
    warm=None
    if initial is not None and 'theta' in initial:
        if (initial['baseline']['variances']!=base['variances'] or initial['baseline']['weights']!=base['weights']):
            raise ValueError('Warm model uses a different analytic baseline')
        with progress.phase('witness_warm_embedding'):
            warm=refine_theta(initial,definition);warm,_=repair(warm,rows,rhs)
    rr=[];cc=[];data=[];b=[]
    radii=[theta_bound]*nvar
    for i,(row,right) in enumerate(zip(rows,rhs)):
        norm=max([abs(v) for v in row.values()]+[F(1,10**30)])
        size=max(abs(right),norm/F(10**6),F(1,10**30))
        for j,v in row.items():
            rr.append(i);cc.append(j);data.append(float(v/size))
            radii[j]=min(radii[j],right/abs(v))
        b.append(float(right/size))
    A=sparse.coo_matrix((data,(rr,cc)),shape=(len(rows),nvar)).tocsr();b=np.asarray(b)
    scales=np.ones(nvar)
    if nvar>200:
        scales=np.array([max(float(r),abs(float(warm[j]))/32 if warm is not None else 0.,1e-30) for j,r in enumerate(radii)])
        A=(A@sparse.diags(scales)).tocsr()
        norms=np.maximum(np.asarray(abs(A).max(axis=1).toarray()).ravel(),np.maximum(abs(b),1e-30))
        A=(sparse.diags(1/norms)@A).tocsr();b=b/norms
    proposal_bounds=np.minimum(float(theta_bound)/scales,1e5)
    nf,df=np.array(list(map(float,n)))*scales,np.array(list(map(float,d)))*scales
    witnesses=[]
    for sign,label in ((1,'minimum'),(-1,'maximum')):
        if label not in endpoints:continue
        # Dinkelbach proposals avoid a poorly scaled Charnes-Cooper variable.
        # Add violated Bernstein constraints lazily; every row is nevertheless
        # checked by the final EXACT contraction, including unselected rows.
        selected=set(range(2*nvar))
        # Consecutive triples cover density/lower/upper constraints together.
        # A uniform stride divisible by three can otherwise omit two families.
        if len(rows)>2*nvar:
            for first in np.linspace(2*nvar,max(2*nvar,len(rows)-3),min(2000,len(rows)-2*nvar),dtype=int):
                selected.update(range(int(first),min(int(first)+3,len(rows))))
        selected.update(range(max(0,len(rows)-2*len(quotes)),len(rows)))
        z=float(nmid/dmid);proposal=np.zeros(nvar);started=time.monotonic();converged=False
        best=np.zeros(nvar);best_value=z;have_feasible=False;anchor=None
        if warm is not None and initial.get('endpoint')==label:
            anchor=warm
            proposal=np.asarray(warm,float)/scales;best=proposal.copy();have_feasible=True
            best_value=(float(nmid)+nf@best)/(float(dmid)+df@best);z=best_value
            progress.event('witness_warm_start','ready',endpoint=label,average=best_value,variables=nvar)
        with progress.phase('witness_'+label+'_optimization',variables=nvar,constraints=len(rows)):
            if nvar<=full_threshold:
                # The full fractional LP is efficient at this size and avoids
                # losing a good witness in intermediate cutting-plane solves.
                norms=np.maximum(np.asarray(abs(A).max(axis=1).toarray()).ravel(),np.maximum(abs(b),1e-30))
                scaled=sparse.diags(1/norms)@A;right=b/norms
                cc=sparse.hstack((scaled,-right[:,None]),format='csr')
                normal=sparse.csr_matrix(np.r_[df,float(dmid)][None,:])
                cc=sparse.vstack((cc,normal,-normal),format='csr')
                limit=float(1/min_approx)
                cost=sign*np.r_[nf,float(nmid)];cost/=max(np.max(abs(cost)),1e-30)
                res=optimize(cost,cc,np.r_[np.zeros(len(b)),1.,-1.],
                    np.r_[-proposal_bounds*limit,0.],
                    np.r_[proposal_bounds*limit,limit],time_limit,
                    solver='simplex' if nvar<=200 else 'ipm')
                if res.x is not None and res.x[-1]>0:
                    proposal=res.x[:-1]/res.x[-1]
                    den=float(dmid)+df@proposal
                    updated=(float(nmid)+nf@proposal)/den if den>0 else sign*np.inf
                    if not have_feasible or sign*updated<sign*best_value:
                        best=proposal.copy();best_value=updated;have_feasible=True
                converged=res.success;selected=set(range(len(rows)))
                iterations=0
            else:
                indices=np.array(sorted(selected),dtype=int)
                engine=WarmLP(np.zeros(nvar),A[indices],b[indices],
                    -proposal_bounds,proposal_bounds)
                if have_feasible:engine.set_primal(best)
                iterations=max_iterations
            for iteration in range(iterations):
                remaining=float(time_limit)-(time.monotonic()-started)
                if remaining<=0:break
                objective=sign*(nf-z*df);objective/=max(float(np.max(np.abs(objective))),1e-30)
                res=engine.solve(objective,remaining)
                if res.x is None:break
                proposal=res.x
                violations=A@proposal-b
                bad=np.flatnonzero(violations>1e-10+1e-8*np.maximum(b,1e-12))
                progress.event('witness_constraint_generation','running',endpoint=label,
                    iteration=iteration+1,active=len(selected),violations=len(bad))
                if len(bad):
                    batch=max(1000,min(12000,8*nvar))
                    take=bad[np.argsort(violations[bad])[-min(batch,len(bad)):]]
                    added=np.array(sorted(set(map(int,take))-selected),dtype=int)
                    if not len(added):break
                    engine.add_rows(A[added],b[added]);selected.update(map(int,added));continue
                den=float(dmid)+df@proposal
                if den<=0:break
                updated=(float(nmid)+nf@proposal)/den
                if not have_feasible or sign*updated<sign*best_value:
                    best=proposal.copy();best_value=updated;have_feasible=True
                if abs(updated-z)<1e-8:converged=True;break
                z=updated
        with progress.phase('witness_exact_primal_repair',endpoint=label):
            theta,factor=repair((best if have_feasible else proposal)*scales,rows,rhs)
        a,denom=average(theta,n,d,density,av,weights,density_floor)
        a=a.intersect(I(amin,amax))
        if anchor is not None:
            aa,ad=average(anchor,n,d,density,av,weights,density_floor);aa=aa.intersect(I(amin,amax))
            if (label=='minimum' and aa.hi<a.hi) or (label=='maximum' and aa.lo>a.lo):
                theta,a,denom,factor=anchor,aa,ad,F(1)
        prices=[]
        for q,(price,coeff) in zip(quotes,quote_values):
            price=price+sum(v*theta[j] for j,v in coeff.items())
            if price.lo<q.bid or price.hi>q.ask:raise ArithmeticError('Repaired witness outside quote interval')
            prices.append(dict(id=q.id,price=price.json(),within_bid_ask=True))
        witnesses.append(dict(endpoint=label,scope='continuous_diffusion',certified=True,
            model='analytic Black mixture plus compact tensor cubic B-spline call correction',
            baseline=base,definition=definition,amin=str(amin),amax=str(amax),
            density_floor=str(density_floor),theta_bound=str(theta_bound),
            subdivisions=subdivisions,quadrature_cells=quadrature_cells,
            theta=list(map(str,theta)),radial_repair_factor=str(factor),
            numerical_search_converged=converged,active_constraints=len(selected),
            numerical_variable_preconditioning=nvar>200,proposal_box_cap=100000 if nvar>200 else None,
            primal_exact_feasible=True,global_dupire_inequalities_verified=True,
            average=a.json(),denominator=denom.json(),quote_prices=prices,
            window=window.json(),quotes=quote_payload(quotes),
            pricing_error='No numerical PDE: the certified call surface is the model price',
            density_integrals=[x.json() for x in density]))
        progress.event('witness_'+label,'certified',average=a.json()['approximate'],repair=float(factor))
    return witnesses


def verify_witness(doc,quotes,window,progress=QUIET,core_cache=None):
    if doc['window']!=window.json() or doc['quotes']!=quote_payload(quotes):
        raise ValueError('Continuous witness linked to different observations or kernel')
    amin,amax=F(doc['amin']),F(doc['amax']);floor=F(doc['density_floor'])
    theta=list(map(F,doc['theta']))
    av=list(map(F,doc['baseline']['variances']));weights=list(map(F,doc['baseline']['weights']))
    if any(w<0 for w in weights) or sum(weights)!=1 or any(not amin<a<amax for a in av):
        raise ValueError('Invalid analytic baseline')
    key=(amin,amax,floor,F(doc['theta_bound']),doc['subdivisions'],tuple(av),tuple(weights),
         tuple(map(F,doc['definition']['time_knots'])),tuple(map(F,doc['definition']['strike_knots'])),
         tuple(theta),tuple(quotes))
    if core_cache is not None and key in core_cache:
        n,d=functional_coefficients(window,doc['definition']);values=core_cache[key]
        progress.event('witness_global_core','reused',coefficients=len(theta))
    else:
        rows,rhs,n,d,values=constraints(quotes,doc['baseline'],window,doc['definition'],amin,amax,
            doc['subdivisions'],floor,F(doc['theta_bound']),progress)
        if len(theta)!=len(n):raise ValueError('Wrong witness coefficient count')
        if any(sum(v*theta[j] for j,v in row.items())>b for row,b in zip(rows,rhs)):
            raise ValueError('Continuous witness fails exact Bernstein constraint')
        if core_cache is not None:core_cache[key]=values
    density=[density_integral(window,a,doc['quadrature_cells'],progress) for a in av]
    if [x.json() for x in density]!=doc['density_integrals']:raise ValueError('Incorrect density integral claim')
    a,denom=average(theta,n,d,density,av,weights,floor);a=a.intersect(I(amin,amax))
    if a.json()!=doc['average'] or denom.json()!=doc['denominator']:
        raise ValueError('Incorrect continuous witness average claim')
    prices=[]
    for q,(price,coeff) in zip(quotes,values):
        price=price+sum(v*theta[j] for j,v in coeff.items())
        if price.lo<q.bid or price.hi>q.ask:raise ValueError('Witness quote containment failed')
        prices.append(dict(id=q.id,price=price.json(),within_bid_ask=True))
    if prices!=doc['quote_prices']:raise ValueError('Incorrect witness price claim')
    return dict(verified=True,continuous_model=True,primal_exact_feasible=True,average=a.json())


def retarget(doc,quotes,window,quadrature_cells=None,progress=QUIET,core_cache=None):
    """Reuse the SAME verified diffusion on another kernel, recomputing A."""
    from .continuous import Window
    if core_cache is None:core_cache={}
    verify_witness(doc,quotes,Window.make(**doc['window']),progress,core_cache)
    if window.json()==doc['window'] and quadrature_cells in (None,doc['quadrature_cells']):return deepcopy(doc)
    result=deepcopy(doc);result['retargeted_from_window']=doc['window'];result['window']=window.json()
    result['quadrature_cells']=quadrature_cells or doc['quadrature_cells']
    av=list(map(F,doc['baseline']['variances']));weights=list(map(F,doc['baseline']['weights']))
    integrals=[density_integral(window,a,result['quadrature_cells'],progress) for a in av]
    n,d=functional_coefficients(window,doc['definition'])
    answer,db=average(list(map(F,doc['theta'])),n,d,integrals,av,weights,F(doc['density_floor']))
    answer=answer.intersect(I(F(doc['amin']),F(doc['amax'])))
    result.update(average=answer.json(),denominator=db.json(),density_integrals=[x.json() for x in integrals],
                  numerical_search_converged=False)
    verify_witness(result,quotes,window,progress,core_cache)
    return result
