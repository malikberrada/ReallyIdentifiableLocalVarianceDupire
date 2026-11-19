"""Rational LP assembly with a proved outward binary relaxation.

All variables are nonnegative. Downward rounding of every row coefficient,
upward rounding of RHS/upper bounds preserves every physical feasible point.
The objective is checked in its original rational form, independently of the
floating objective used by HiGHS to propose multipliers.
"""
from fractions import Fraction as F
from pathlib import Path
import json
import math
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from .lp import Polytope,solve,exact_dual_lower
from .io import save_json,sha256
from .progress import QUIET
from .native_solver import optimize
from .display import approximate


def q(x):return x if isinstance(x,F) else F.from_float(float(x))


def down(x):
    x=F(x);v=float(x)
    if not math.isfinite(v):raise ValueError('Nonfinite LP coefficient')
    return np.nextafter(v,-np.inf) if F.from_float(v)>x else v


def up(x):return -down(-F(x))


class RationalLP:
    def __init__(self):
        self.upper=[];self.names=[];self.rows=[];self.rhs=[];self.labels=[]

    def var(self,upper,name=''):
        if F(upper)<0:raise ValueError('Negative variable upper bound')
        j=len(self.upper);self.upper.append(F(upper));self.names.append(name)
        return j

    def row(self,entries,rhs=0,label=''):
        pairs=entries.items() if isinstance(entries,dict) else entries
        combined={}
        for j,v in pairs:
            v=F(v)
            if v:combined[j]=combined.get(j,F(0))+v
        self.rows.append({j:v for j,v in combined.items() if v})
        self.rhs.append(F(rhs));self.labels.append(label)

    def eq(self,entries,rhs=0,label=''):
        pairs=list(entries.items() if isinstance(entries,dict) else entries)
        self.row(pairs,rhs,label)
        self.row([(j,-v) for j,v in pairs],-F(rhs),label)

    def binary(self):
        rr=[];cc=[];dd=[]
        for i,row in enumerate(self.rows):
            scale=max([abs(v) for v in row.values()]+[abs(self.rhs[i]),F(1,10**30)])
            for j,v in sorted(row.items()):rr.append(i);cc.append(j);dd.append(down(v/scale))
        A=sparse.coo_matrix((dd,(rr,cc)),shape=(len(self.rows),len(self.upper))).tocsr()
        rhs=[up(b/max([abs(v) for v in row.values()]+[abs(b),F(1,10**30)])) for row,b in zip(self.rows,self.rhs)]
        return Polytope(A,np.array(rhs),np.zeros(len(self.upper)),
                        np.array([up(x) for x in self.upper]),None,None,self.labels)

    def check_binary_relaxation(self,p):
        if p.A.shape!=(len(self.rows),len(self.upper)):raise ValueError('Shape mismatch')
        for i,row in enumerate(self.rows):
            scale=max([abs(v) for v in row.values()]+[abs(self.rhs[i]),F(1,10**30)])
            actual={int(p.A.indices[z]):q(p.A.data[z]) for z in range(p.A.indptr[i],p.A.indptr[i+1])}
            if set(actual)!=set(row) or any(actual[j]>v/scale for j,v in row.items()):
                raise ValueError('LP coefficients exclude possible continuous projections')
            if q(p.b[i])<self.rhs[i]/scale:raise ValueError('RHS rounded inward')
        if any(l!=0 for l in p.lo) or any(q(v)<u for v,u in zip(p.hi,self.upper)):
            raise ValueError('Variable bounds rounded inward')
        return True


def certificate(path,p,objective,res,progress=QUIET):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    objective=list(map(F,objective));lam=np.maximum(0,-np.asarray(res.ineqlin.marginals))
    with progress.phase('continuous_dual_exact',certificate=path.stem):
        value=exact_dual_lower(p.A,p.b,p.lo,p.hi,objective,lam,progress)
    payload=path.with_suffix('.npz')
    np.savez_compressed(payload,data=p.A.data,indices=p.A.indices,indptr=p.A.indptr,
        shape=p.A.shape,b=p.b,lo=p.lo,hi=p.hi,objective=np.array([str(x) for x in objective]),lam=lam)
    result=dict(scope='outward_relaxation_of_continuous_models',payload=payload.name,
                sha256=sha256(payload),lower_bound=str(value),approximate=approximate(value),
                exact_objective=True,primal_claim='No physical model inferred from the outer LP primal')
    save_json(path,result)
    return value


def verify(path,progress=QUIET):
    path=Path(path);doc=json.loads(path.read_text(encoding='utf-8'));payload=path.parent/doc['payload']
    if sha256(payload)!=doc['sha256']:raise ValueError('Continuous LP payload changed')
    with np.load(payload,allow_pickle=False) as z:
        A=sparse.csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(z['shape']))
        if not all(np.all(np.isfinite(z[k])) for k in ('data','b','lo','hi','lam')):
            raise ValueError('Nonfinite certificate')
        if (len(z['b'])!=A.shape[0] or len(z['lam'])!=A.shape[0] or
            len(z['lo'])!=A.shape[1] or len(z['hi'])!=A.shape[1] or
            len(z['objective'])!=A.shape[1] or np.any(z['lo']>z['hi'])):
            raise ValueError('Invalid certificate dimensions or variable bounds')
        p=Polytope(A,z['b'].copy(),z['lo'].copy(),z['hi'].copy(),None,None,[])
        objective=[F(x) for x in z['objective']]
        value=exact_dual_lower(A,p.b,p.lo,p.hi,objective,z['lam'],progress)
    if value!=F(doc['lower_bound']):raise ValueError('Incorrect exact dual claim')
    return p,objective,value


def fractional_outer(builder,n,d,directory,progress=QUIET,time_limit=120,denominator_row=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    p=builder.binary();builder.check_binary_relaxation(p)
    r=optimize(list(map(float,d)),p.A,p.b,p.lo,p.hi,time_limit,progress=progress,label='continuous_denominator')
    if not r.dual_finite and denominator_row is None:raise RuntimeError('No finite dual proposal: '+r.message)
    numerical=None
    if r.dual_finite:
        numerical=certificate(directory/('denominator_numerical.json' if denominator_row is not None else 'denominator.json'),p,d,r,progress)
    chosen=r
    source='occupation_dual'
    if denominator_row is not None:
        # A separately VERIFIED continuous subsolution justifies this row.
        # Its dyadic RHS and -1 coefficients give an exact one-row dual, so
        # tiny numerical stationarity errors cannot destroy strict positivity.
        lam=np.zeros(p.A.shape[0]);lam[denominator_row]=1.
        analytic=SimpleNamespace(ineqlin=SimpleNamespace(marginals=-lam))
        direct=exact_dual_lower(p.A,p.b,p.lo,p.hi,d,lam,progress)
        if numerical is None or direct>numerical:chosen=analytic;source='verified_continuous_floor_row'
        dmin=certificate(directory/'denominator.json',p,d,chosen,progress)
    else:dmin=numerical
    result=dict(denominator_lower=str(dmin),variables=len(p.lo),constraints=p.A.shape[0])
    if denominator_row is not None:result.update(denominator_source=source,numerical_denominator_lower=str(numerical) if numerical is not None else None)
    if dmin<=0:
        return dict(**result,status='denominator_unresolved')
    if denominator_row is not None and dmin<F(1,10**8):
        return dict(**result,status='positive_denominator_normalization_ill_conditioned')
    # The physical Charnes-Cooper transform satisfies the exact rational rows
    # below. A second componentwise outward conversion preserves that transform.
    cc=fractional_builder(p,d,dmin)
    cp=cc.binary();cc.check_binary_relaxation(cp)
    answers=[]
    for sign,name in ((1,'minimum'),(-1,'maximum')):
        f=[sign*v for v in n]+[F(0)]
        res=optimize(list(map(float,f)),cp.A,cp.b,cp.lo,cp.hi,time_limit,progress=progress,label='continuous_'+name)
        if not res.dual_finite:raise RuntimeError('No finite dual proposal: '+res.message)
        val=certificate(directory/(name+'.json'),cp,f,res,progress)
        answers.append(val if sign==1 else -val)
    return dict(**result,status='continuous_outer_duals_verified',lower=str(answers[0]),
                upper=str(answers[1]),approximate=list(map(approximate,answers)))


def fractional_builder(p,d,dmin):
    cc=RationalLP();smax=1/dmin
    for j,u in enumerate(p.hi):cc.var(q(u)*smax,f'y{j}')
    scale=cc.var(smax,'scale')
    for i in range(p.A.shape[0]):
        row={int(p.A.indices[z]):q(p.A.data[z]) for z in range(p.A.indptr[i],p.A.indptr[i+1])}
        row[scale]=-q(p.b[i]);cc.row(row,0,'projected_constraint')
    for j,u in enumerate(p.hi):cc.row({j:1,scale:-q(u)},0,'scaled_upper')
    cc.eq({j:v for j,v in enumerate(d) if v},1,'denominator_normalization')
    return cc
