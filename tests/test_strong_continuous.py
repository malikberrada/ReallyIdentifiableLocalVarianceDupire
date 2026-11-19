"""Independent continuous-model checks, not just solver residual checks."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
from math import comb
from pathlib import Path
import json
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from dupire_identify import certlp,inner,occupation,time_models,tightening
from dupire_identify.continuous import Window
from dupire_identify.curves import spline_piece,value
from dupire_identify.density import gamma_box,density_integral
from dupire_identify.interval import black_call
from dupire_identify.io import Quote,write_quotes
from dupire_identify.models import ContinuousModel
from dupire_identify.native_solver import WarmLP
from dupire_identify.polynomial import deriv,bump


def observations(a=F(1,25)):
    out=[]
    for i,t in enumerate((F(1,4),F(3,4))):
        for j,k in enumerate((F(9,10),F(1),F(11,10))):
            p=black_call(k,a*t)
            out.append(Quote(f'{i}_{j}',t,k,max(F(0),p.lo-F(1,1000)),p.hi+F(1,1000)))
    return out


W=Window.make('3/8',1,'1/16','1/10')
AMIN,AMAX=F(1,400),F(1,4)


def test_componentwise_outward_rounding_and_tamper():
    b=certlp.RationalLP();b.var(F(1,3));b.var(F(5,7))
    b.row({0:F(2,3),1:F(-1,7)},F(1,5));p=b.binary()
    assert b.check_binary_relaxation(p)
    p.A.data[0]=np.nextafter(p.A.data[0],np.inf)
    with pytest.raises(ValueError,match='exclude'):b.check_binary_relaxation(p)


def test_known_fractional_problem_exact_duals(tmp_path):
    b=certlp.RationalLP();b.var(1);b.var(1);b.eq({0:1,1:1},1)
    # (x+2y)/(x+3y) lies in [2/3,1]. Physical objectives remain rational.
    r=certlp.fractional_outer(b,[F(1),F(2)],[F(1),F(3)],tmp_path,time_limit=10)
    assert F(r['lower'])<=F(2,3) and float(F(r['lower']))>2/3-1e-10
    assert F(r['upper'])>=1 and float(F(r['upper']))<1+1e-10
    for name in ('denominator','minimum','maximum'):certlp.verify(tmp_path/(name+'.json'))
    path=tmp_path/'minimum.json';doc=json.loads(path.read_text());doc['lower_bound']='1'
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='dual claim'):certlp.verify(path)


def test_warm_simplex_added_rows_and_changed_objective():
    engine=WarmLP([-1.,0.],[[1.,1.]],[1.],[0.,0.],[2.,2.])
    r=engine.solve([-1.,0.]);assert r.success and abs(r.x[0]-1)<1e-10
    engine.add_rows([[1.,0.]],[.3]);r=engine.solve([-1.,-1.])
    assert r.success and r.x[0]<=.3+1e-10 and abs(sum(r.x)-1)<1e-10
    assert len(r.ineqlin.marginals)==2


def independent_g(t,k,a=.04):
    v=a*t;d=(-np.log(k)-v/2)/np.sqrt(v)
    return k*np.exp(-d*d/2)/np.sqrt(2*np.pi*v)


def test_density_box_and_hessian_against_independent_differences():
    box=gamma_box(F(39,100),F(41,100),F(98,100),F(102,100),F(1,25))
    t,k=.4,1.;h=1e-5;g=independent_g(t,k)
    gt=(independent_g(t+h,k)-independent_g(t-h,k))/(2*h)
    gk=(independent_g(t,k+h)-independent_g(t,k-h))/(2*h)
    gtt=(independent_g(t+h,k)-2*g+independent_g(t-h,k))/h**2
    gkk=(independent_g(t,k+h)-2*g+independent_g(t,k-h))/h**2
    gtk=(independent_g(t+h,k+h)-independent_g(t+h,k-h)-independent_g(t-h,k+h)+independent_g(t-h,k-h))/(4*h*h)
    for bound,number in zip(box,(g,gt,gk,gtt,gtk,gkk)):
        assert float(bound.lo)<=number<=float(bound.hi)


def test_certified_quadrature_contains_independent_integral():
    def f(t,k):
        return (1-((t-.375)/.0625)**2)**3*(1-((k-1)/.1)**2)**3*independent_g(t,k)
    actual=quad(lambda t:quad(lambda k:f(t,k),.9,1.1,epsabs=1e-13)[0],.3125,.4375,epsabs=1e-13)[0]
    coarse=density_integral(W,F(1,25),8);fine=density_integral(W,F(1,25),16)
    assert float(coarse.lo)<actual<float(coarse.hi)
    assert float(fine.lo)<actual<float(fine.hi)
    assert fine.hi-fine.lo<coarse.hi-coarse.lo


def test_compact_cubic_splines_have_exact_C2_joins():
    knots=tuple(F(1,4)+F(i,16) for i in range(9))
    for index in range(len(knots)-4):
        for cell,x in enumerate(knots):
            left=list(spline_piece(knots,index,3,cell-1)) if cell else [F(0)]
            right=list(spline_piece(knots,index,3,cell)) if cell<len(knots)-1 else [F(0)]
            for order in range(3):
                assert value(left,x)==value(right,x)
                left,right=deriv(left),deriv(right)


def test_exact_time_model_attains_lower_bound_and_preserves_prices():
    q=observations();models=time_models.candidates(q,W,F(1,25),AMIN,AMAX)
    assert len(models)==2 and F(models[0]['average']['hi'])==AMIN
    for doc in models:
        assert time_models.verify(doc,q,W)['continuous_model']
        model=ContinuousModel(doc)
        for quote in q:
            assert time_models.integrated_variance(doc,quote.t)==F(1,25)*quote.t
            assert model.call(quote.t,quote.k).lo>=quote.bid
        assert model.local_variance(W.t,W.k).lo==F(doc['target_variance'])
    bad=deepcopy(models[0]);bad['segments'][1][2][0]='1/100000'
    with pytest.raises(ValueError):time_models.verify(bad,q,W)


def test_spline_witness_exact_primal_and_reverification():
    q=observations();base,_=inner.baseline(q,[F(1,25)],AMIN,AMAX)
    definition=inner.spec(W,q,6,6)
    models=inner.search(q,W,AMIN,AMAX,base,definition,quadrature_cells=12,time_limit=15)
    for doc in models:
        assert inner.verify_witness(doc,q,W)['primal_exact_feasible']
        model=ContinuousModel(doc)
        for t in (F(1,3),F(3,8),F(2,5)):
            for k in (F(9,10),F(1),F(11,10)):
                a=model.local_variance(t,k);assert AMIN<=a.lo<=a.hi<=AMAX
        bad=deepcopy(doc);bad['theta'][0]='100000'
        with pytest.raises(ValueError,match='Bernstein'):inner.verify_witness(bad,q,W)


def truncated_moment(t,left,right,order,a=.04):
    v=a*t;s=np.sqrt(v)
    def cumulative(x):
        if x==0:return 0.
        if np.isinf(x):return 1.
        return ndtr((np.log(x)+(.5-order)*v)/s)
    return np.exp(order*(order-1)*v/2)*(cumulative(right)-cumulative(left))


@pytest.mark.parametrize('degree',[1,2])
def test_independent_lognormal_projection_satisfies_continuous_relaxation(degree):
    q=observations();b,n,d,definition=occupation.assemble(q,W,AMIN,AMAX,time_cells=2,
        strike_cells=4,target_cells=4,time_degree=degree,bs_bounds=False)
    ts=np.array([float(F(x)) for x in definition['times']]);ks=np.array([float(F(x)) for x in definition['strikes']])
    nodes,weights=np.polynomial.legendre.leggauss(64);pk=bump(W.k,W.hk)
    moments={};weighted={}
    for it,(ta,tb) in enumerate(zip(ts[:-1],ts[1:])):
        time=ta+(tb-ta)*(nodes+1)/2;u=(nodes+1)/2
        for r in range(degree+1):
            wt=weights*(tb-ta)/2*comb(degree,r)*u**r*(1-u)**(degree-r)
            for j,left in enumerate(ks):
                right=ks[j+1] if j+1<len(ks) else np.inf
                moments[it,r,j]=[sum(wt[z]*truncated_moment(t,left,right,o) for z,t in enumerate(time)) for o in range(3)]
                if ta>=float(W.t-W.ht) and tb<=float(W.t+W.ht) and left>=float(W.k-W.hk) and right<=float(W.k+W.hk):
                    def weighted_moment(t):
                        return sum(float(co)*truncated_moment(t,left,right,o+2) for o,co in enumerate(pk))
                    weighted[it,r,j]=sum(wt[z]*(1-((t-float(W.t))/float(W.ht))**2)**3*weighted_moment(t) for z,t in enumerate(time))
    x=np.zeros(len(b.upper))
    for index,name in enumerate(b.names):
        parts=name.split('_')
        if parts[0]=='term':
            it,j,order=map(int,parts[1:]);right=ks[j+1] if j+1<len(ks) else np.inf
            x[index]=truncated_moment(ts[it],ks[j],right,order)
        elif parts[0]=='occ':
            it,r,j,order=map(int,parts[1:]);x[index]=moments[it,r,j][order]
        elif parts[0]=='flux':
            it,r,j=map(int,parts[1:]);x[index]=.04*moments[it,r,j][2]
        else:
            it,r,j=map(int,parts[2:]);x[index]=weighted[it,r,j]*(.04 if parts[1]=='flux' else 1.)
    p=b.binary();assert b.check_binary_relaxation(p)
    assert np.max(p.A@x-p.b)<2e-9
    assert np.min(x)>-2e-10 and np.max(x-p.hi)<2e-10
    assert abs(np.asarray(n,float)@x-.04*(np.asarray(d,float)@x))<1e-12
    assert 'ito_black_flux_budget' in b.labels


def test_tighten_verify_resume_and_stale_files(tmp_path,monkeypatch):
    q=observations();q[0]=replace(q[0],id='côté_irrégulier')
    path=tmp_path/'quotes.csv';write_quotes(path,q)
    out=tmp_path/'run'
    kwargs=dict(window=W,outer_levels=(),spline_levels=(6,),subdivisions=1,
                quadrature_cells=12,time_limit=15,progress_mode='off')
    result=tightening.run(path,out,**kwargs)
    # Files left by a previous, longer refinement must not enter this proof.
    (out/'inner_99.json').write_text('unrelated old data')
    def forbidden(*args,**kwargs):raise AssertionError('Verifier restarted optimizer')
    monkeypatch.setattr(inner,'optimize',forbidden);monkeypatch.setattr(certlp,'optimize',forbidden)
    original_read=Path.read_text
    def windows_default(self,*args,**kwargs):
        kwargs.setdefault('encoding','cp1252')
        return original_read(self,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',windows_default)
    checked=tightening.verify(out)
    assert checked['verified'] and not checked['optimizer_restarted']
    resumed=tightening.run(path,out,resume=True,**kwargs)
    assert resumed['gap']==result['gap']
    (out/'minimum_model.json').write_text('{}')
    with pytest.raises(ValueError,match='Changed artifact'):tightening.verify(out)


def test_continuous_flux_budget_actually_strengthens_bounds(tmp_path):
    # Designed informative case: quotes almost saturate the announced amax.
    # This is a synthetic sharpness check, not a claim about market smiles.
    from dupire_identify.workflow import synthetic_quotes
    path=tmp_path/'quotes.csv'
    synthetic_quotes(path,a0=F(249999999,10**9),spread=F(1,10**10))
    result=tightening.run(path,tmp_path/'run',window=Window.make('.5',1,'.24','.65'),
        outer_levels=(4,),spline_levels=(6,),baseline_variances=(F(249999999,10**9),),
        quadrature_cells=16,time_limit=60,progress_mode='off')
    assert F(result['outer']['lower'])>F(2499,10000)
    assert F(result['outer']['upper'])==AMAX
    assert result['gap']['endpoints_within_tolerance_certified']
    checked=tightening.verify(tmp_path/'run')
    assert checked['verified'] and checked['denominator_positive_proved']
