"""Sparse finite-grid identification LP, with exact post-verified dual bounds.

Scope: the stored binary-coefficient finite LP, never the continuous class.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from scipy.optimize import linprog,OptimizeWarning
import warnings
from .continuous import Window
from .io import save_json, sha256
from .progress import QUIET
from .display import approximate


@dataclass
class Polytope:
    A: sparse.csr_matrix
    b: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    times: np.ndarray
    strikes: np.ndarray
    labels: list


def build_polytope(quotes, amin, amax, nt=16, nk=35, kmax=2.5, progress=QUIET):
    amin,amax=float(amin),float(amax)
    if not 0<amin<=amax or nt<2 or nk<5: raise ValueError("Invalid grid/variance bounds")
    tmax=max(float(q.t) for q in quotes)
    kmax=max(float(kmax),max(float(q.k) for q in quotes)*1.1)
    times=np.unique(np.r_[np.linspace(0,tmax,nt),[float(q.t) for q in quotes]])
    strikes=np.unique(np.r_[np.linspace(0,kmax,nk),1,[float(q.k) for q in quotes]])
    m,n=len(times),len(strikes)
    lower=np.tile(np.maximum(1-strikes,0),m)
    upper=np.ones(m*n)
    lower[:n]=upper[:n]=np.maximum(1-strikes,0)
    # c(T,0)=1; right boundary deliberately remains free in [intrinsic,1].
    for i in range(m): lower[i*n]=upper[i*n]=1
    rows,cols,vals,b,labels=[],[],[],[],[]
    def row(entries,rhs,label):
        r=len(b)
        for j,x in entries:
            if x: rows.append(r);cols.append(j);vals.append(x)
        b.append(rhs);labels.append(label)
    for i in range(m):
        for j in range(n-1):
            h=strikes[j+1]-strikes[j]
            row([(i*n+j+1,1),(i*n+j,-1)],0,"strike_decreasing")
            row([(i*n+j,1),(i*n+j+1,-1)],h,"strike_lipschitz")
        for j in range(1,n-1):
            dl=strikes[j]-strikes[j-1];dr=strikes[j+1]-strikes[j]
            g=[(i*n+j-1,2/(dl*(dl+dr))),
               (i*n+j,-2/(dl*dr)),(i*n+j+1,2/(dr*(dl+dr)))]
            row([(c,-v) for c,v in g],0,"convexity")
            if i:
                dt=times[i]-times[i-1]
                time=[(i*n+j,2/dt),((i-1)*n+j,-2/dt)]
                row(time+[(c,-amax*strikes[j]**2*v) for c,v in g],0,"dupire_upper_implicit")
                row([(c,-v) for c,v in time]+[(c,amin*strikes[j]**2*v) for c,v in g],0,"dupire_lower_implicit")
        if i:
            for j in range(n): row([((i-1)*n+j,1),(i*n+j,-1)],0,"calendar")
        progress.event("sparse_assembly","running",done=i+1,total=m)
    for q in quotes:
        i=int(np.searchsorted(times,float(q.t)));j=int(np.searchsorted(strikes,float(q.k)))
        row([(i*n+j,1)],float(q.ask),"ask:"+q.id)
        row([(i*n+j,-1)],-float(q.bid),"bid:"+q.id)
    A=sparse.coo_matrix((vals,(rows,cols)),shape=(len(b),m*n)).tocsr()
    A.sum_duplicates();A.eliminate_zeros()
    return Polytope(A,np.asarray(b),lower,upper,times,strikes,labels)


def solve(p, objective, progress=QUIET, label="LP", time_limit=120, method="highs",extra_options=None):
    with progress.phase(label,variables=len(objective),constraints=p.A.shape[0],
                        solver="HiGHS C++",time_limit_s=time_limit):
        options=dict(time_limit=float(time_limit),primal_feasibility_tolerance=1e-9,
                     dual_feasibility_tolerance=1e-9)
        options.update(extra_options or {})
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected:.*run_crossover',category=OptimizeWarning)
            res=linprog(objective,A_ub=p.A,b_ub=p.b,
                        bounds=np.column_stack([p.lo,p.hi]),method=method,options=options)
    if not res.success:
        raise RuntimeError(f"{label}: status={res.status}; {res.message}. "
                           "Numerical solver status is not an exact infeasibility certificate.")
    progress.event(label,"optimal",iterations=int(res.nit),objective=float(res.fun))
    return res


def exact_dual_lower(A,b,lo,hi,f,lam,progress=QUIET):
    """For Ax<=b and l<=x<=u, ANY lambda>=0 yields
    min f.x >= -lambda.b + sum_j min(r_j*l_j,r_j*u_j), r=f+A'lambda.
    All operations below are exact on stored float64 binary coefficients.
    No small stationarity residual is discarded.
    """
    A=A.tocsr()
    convert=lambda v:v if isinstance(v,F) else F.from_float(float(v))
    lam=[max(F(0),convert(v)) for v in lam]
    r=[convert(v) for v in f]
    value=F(0)
    for i,l in enumerate(lam):
        if l:
            value-=l*convert(b[i])
            for p in range(A.indptr[i],A.indptr[i+1]):
                r[A.indices[p]]+=l*convert(A.data[p])
        if (i+1)%1000==0: progress.event("exact_dual_verification","running",done=i+1,total=A.shape[0])
    correction=sum(min(v*convert(a),v*convert(bj)) for v,a,bj in zip(r,lo,hi))
    return value+correction


def exact_primal_violation(A,b,lo,hi,x):
    cv=lambda v:F.from_float(float(v))
    x=list(map(cv,x))
    violation=max([F(0)]+[cv(l)-v for l,v in zip(lo,x)]+[v-cv(u) for u,v in zip(hi,x)])
    for i in range(A.shape[0]):
        lhs=sum(cv(A.data[j])*x[A.indices[j]] for j in range(A.indptr[i],A.indptr[i+1]))
        violation=max(violation,lhs-cv(b[i]))
    return violation


def save_certificate(directory,name,p,f,res,progress=QUIET,scope="finite_binary_lp"):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    binary=directory/(name+".npz")
    lam=np.maximum(0,-np.asarray(res.ineqlin.marginals))
    np.savez_compressed(binary,data=p.A.data,indices=p.A.indices,indptr=p.A.indptr,
                        shape=p.A.shape,b=p.b,lo=p.lo,hi=p.hi,f=f,lam=lam,x=res.x)
    with progress.phase("exact_dual_verification",certificate=name):
        bound=exact_dual_lower(p.A,p.b,p.lo,p.hi,f,lam,progress)
        violation=exact_primal_violation(p.A,p.b,p.lo,p.hi,res.x)
    doc=dict(scope=scope,payload=binary.name,sha256=sha256(binary),
             lower_bound=str(bound),approximate=approximate(bound),
             primal_exact_feasible=violation==0,primal_violation=str(violation),
             numerical_objective=float(res.fun),
             dual_formula="-lambda*b + sum min((f+A^T*lambda)_j*l_j,(f+A^T*lambda)_j*u_j)")
    save_json(directory/(name+".json"),doc)
    return bound,doc


def verify_certificate(path,progress=QUIET):
    path=Path(path);doc=json.loads(path.read_text(encoding="utf-8"))
    payload=path.parent/doc["payload"]
    if sha256(payload)!=doc["sha256"]: raise ValueError("Certificate payload checksum mismatch")
    with np.load(payload,allow_pickle=False) as z:
        A=sparse.csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(z['shape']))
        if not all(np.all(np.isfinite(z[k])) for k in ("data","b","lo","hi","f","lam","x")):
            raise ValueError("Nonfinite certificate data")
        if np.any(z['lo']>z['hi']): raise ValueError("Invalid variable bounds")
        bound=exact_dual_lower(A,z['b'],z['lo'],z['hi'],z['f'],z['lam'],progress)
        violation=exact_primal_violation(A,z['b'],z['lo'],z['hi'],z['x'])
    if bound != F(doc["lower_bound"]): raise ValueError("Incorrect claimed dual bound")
    if (violation==0)!=doc["primal_exact_feasible"] or violation!=F(doc["primal_violation"]):
        raise ValueError("Incorrect primal feasibility claim")
    return dict(verified=True,scope=doc["scope"],lower_bound=str(bound),
                primal_exact_feasible=violation==0)


def functionals(p,w):
    t,k=p.times,p.strikes
    n=np.zeros(len(p.lo));d=n.copy()
    for i in range(1,len(t)):
        mid=(t[i]+t[i-1])/2
        wt=max(0,1-((mid-float(w.t))/float(w.ht))**2)**3
        for j in range(1,len(k)-1):
            wk=max(0,1-((k[j]-float(w.k))/float(w.hk))**2)**3
            weight=wt*wk
            if not weight: continue
            dt=t[i]-t[i-1];dl=k[j]-k[j-1];dr=k[j+1]-k[j]
            volume=dt*(dl+dr)/2*weight
            n[i*len(k)+j]+=volume*2/dt
            n[(i-1)*len(k)+j]-=volume*2/dt
            d[i*len(k)+j-1]+=volume*k[j]**2*2/(dl*(dl+dr))
            d[i*len(k)+j]-=volume*k[j]**2*2/(dl*dr)
            d[i*len(k)+j+1]+=volume*k[j]**2*2/(dr*(dl+dr))
    if not np.any(d): raise ValueError("Kernel contains no grid stencil; refine nt/nk")
    return n,d


def identify_discrete(p,window,output,progress=QUIET,time_limit=120):
    n,d=functionals(p,window)
    rmin=solve(p,d,progress,"denominator_min",time_limit)
    dmin,cert=save_certificate(output,"denominator",p,d,rmin,progress)
    if dmin<=0:
        return dict(scope="finite_grid_only",status="denominator_not_certified_positive",
                    denominator_certificate=cert,continuous_bound=False)
    # Charnes-Cooper: y=x*s, d.y=1, s=1/(d.x). Finite box
    # upper bounds for s/y are rounded upward from the verified dmin.
    smax=np.nextafter(float(1/dmin),np.inf)
    if not np.isfinite(smax) or smax>1e12: raise ValueError("Denominator too small for stable fractional LP")
    count=len(n)
    A=sparse.hstack([p.A,-sparse.csr_matrix(p.b[:,None])],format="csr")
    identity=sparse.eye(count,format="csr")
    A=sparse.vstack([A,sparse.hstack([identity,-sparse.csr_matrix(p.hi[:,None])]),
                     sparse.hstack([-identity,sparse.csr_matrix(p.lo[:,None])]),
                     sparse.csr_matrix(np.r_[d,0][None,:]),
                     sparse.csr_matrix(-np.r_[d,0][None,:])],format="csr")
    b=np.r_[np.zeros(p.A.shape[0]+2*count),1,-1]
    cc=Polytope(A,b,np.zeros(count+1),np.r_[np.full(count,smax),smax],p.times,p.strikes,[])
    extrema=[]
    certs=[]
    for sign,name in [(1,"fractional_min"),(-1,"fractional_max")]:
        f=np.r_[sign*n,0]
        res=solve(cc,f,progress,name,time_limit)
        bound,certificate=save_certificate(output,name,cc,f,res,progress)
        x=res.x[:-1]/res.x[-1]
        np.savez_compressed(Path(output)/(name+"_surface.npz"),times=p.times,strikes=p.strikes,
                            calls=x.reshape(len(p.times),len(p.strikes)))
        extrema.append(bound if sign==1 else -bound);certs.append(certificate)
    return dict(scope="finite_grid_only",status="finite_lp_bounds_verified",
                lower=str(extrema[0]),upper=str(extrema[1]),
                approximate=[float(v) for v in extrema],
                denominator_lower=str(dmin),certificates=certs,
                numerical_surfaces_are_continuous_witnesses=False,
                continuous_bound=False,variables=len(p.lo),constraints=p.A.shape[0],
                truncation="right boundary free; finite stencils are still not a continuous relaxation")
