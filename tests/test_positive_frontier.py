"""Positive global floors, exact static repairs, and honest frontier closure."""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import json
import numpy as np
import pytest
from scipy.interpolate import BSpline
from dupire_identify import denominator,certlp,inner,time_models,tightening,certified_frontier
from dupire_identify.continuous import Window
from dupire_identify.density import density_integral
from dupire_identify.interval import black_call
from dupire_identify.io import Quote,write_quotes,save_json,sha256
from dupire_identify.workflow import synthetic_quotes


W=Window.make('3/8',1,'1/16','1/10');LO,HI=F(1,400),F(1,4)


def quotes():
    result=[]
    for t in (F(1,4),F(3,4)):
        for k in (F(9,10),F(1),F(11,10)):
            p=black_call(k,F(1,25)*t)
            result.append(Quote(str(t)+'_'+str(k),t,k,max(F(0),p.lo-F(1,1000)),p.hi+F(1,1000)))
    return result


def test_brownian_floor_is_positive_and_below_independent_models():
    doc=denominator.brownian_floor(W,LO,HI);lower=denominator.verify_brownian(doc,W,LO,HI)
    assert lower>0 and lower==F(1,2**39)
    for a in (LO,F(1,25),HI):
        actual=density_integral(W,a,16)
        assert lower<actual.lo
    bad=deepcopy(doc);bad['power_two_exponent']-=1
    with pytest.raises(ValueError,match='Brownian'):denominator.verify_brownian(bad,W,LO,HI)


def test_positivity_never_assumes_an_initial_state_at_later_time():
    with pytest.raises(ValueError,match='positive'):denominator.brownian_floor(W,0,HI)
    config=denominator.definition(W,4,4,start=F(1,4))
    n=(len(config['time_knots'])-4)*(len(config['strike_knots'])-4)
    doc=dict(window=W.json(),amin=str(LO),amax=str(HI),definition=config,subdivisions=1,
             theta=['0']*n,lower_bound='0',epsilon='0',initial_spline_value='0',positive=False,dyadic_lower='0')
    with pytest.raises(ValueError,match='time zero'):denominator.verify(doc,W,LO,HI)


def test_linked_dyadic_floor_survives_exact_dual_verification(tmp_path):
    lp=certlp.RationalLP();lp.var(1);lp.var(1)
    lp.row({0:-1,1:LO});lp.row({0:1,1:-HI})
    floor=F(1,8192);floor_row=len(lp.rows);lp.row({1:-1},-floor)
    r=certlp.fractional_outer(lp,[F(1),F(0)],[F(0),F(1)],tmp_path,time_limit=10,denominator_row=floor_row)
    _,_,proved=certlp.verify(tmp_path/'denominator.json')
    assert proved>=floor and F(r['denominator_lower'])==proved
    assert F(r['lower'])<=LO and F(r['upper'])>=HI


def scipy_basis(knots,x,order=0):
    out=[]
    for i in range(len(knots)-4):
        b=BSpline.basis_element(knots[i:i+5],extrapolate=False)
        if order:b=b.derivative(order)
        v=b(x);out.append(0. if np.isnan(v) else float(v))
    return np.array(out)


def test_static_subsolution_is_useful_and_dominates_its_portfolio(tmp_path):
    q=synthetic_quotes(tmp_path/'quotes.csv')
    doc=denominator.search_static(q,W,LO,HI,8,10,time_limit=30)
    floor=denominator.verify_static(doc,q,W,LO,HI)
    assert floor>=F(1,16384)
    assert F(doc['lower_bound'])<=density_integral(W,F(1,25),16).lo
    config=doc['definition'];ts=list(map(lambda x:float(F(x)),config['time_knots']))
    ks=list(map(lambda x:float(F(x)),config['strike_knots']))
    theta=np.asarray(list(map(lambda x:float(F(x)),doc['theta']))).reshape(len(ts)-4,len(ks)-4)
    start=float(F(config['start']));horizon=float(F(config['horizon']));eps=float(F(doc['epsilon']))
    port=doc['portfolio'];lam=list(map(lambda x:float(F(x)),port['weights']))
    used={z['id']:z for z in doc['quotes']}
    strikes=[float(F(z['k'])) for z in doc['quotes']]
    for x in np.r_[np.linspace(0,3,61),10.,100.]:
        p=0. if x<=ks[0] or x>=ks[-1] else scipy_basis(ts,start)@theta@scipy_basis(ks,x)
        g=float(F(port['cash']))+float(F(port['stock']))*x+sum(l*max(0,x-k) for l,k in zip(lam,strikes))
        assert g<=p-eps*(horizon-start)+1e-10
    for t in np.linspace(start,horizon,9)[:-1]:
        for x in np.linspace(ks[0]+.001,ks[-1]-.001,19):
            bt=scipy_basis(ts,t);dt=scipy_basis(ts,t,1);bk=scipy_basis(ks,x);ddk=scipy_basis(ks,x,2)
            phi=max(0,1-((t-.375)/.0625)**2)**3*max(0,1-((x-1)/.1)**2)**3
            for a in (float(LO),float(HI)):
                residual=dt@theta@bk+eps+.5*a*x*x*(bt@theta@ddk)+phi*x*x
                assert residual>=-1e-8
    bad=deepcopy(doc);bad['portfolio']['cash']=str(F(bad['portfolio']['cash'])+F(1,100))
    with pytest.raises(ValueError,match='static'):denominator.verify_static(bad,q,W,LO,HI)


def test_exact_knot_insertion_keeps_continuous_model_unchanged():
    old=[F(i,8) for i in range(9)];c=[F(i-2,9) for i in range(5)]
    new=sorted(set(old)|{F(-1,4),F(-1,8),F(1,16),F(3,16),F(5,16),F(9,8)})
    mapped=inner.refine_curve(old,c,new)
    for j in range(-5,40):
        x=F(j,32)
        for order in range(3):
            assert sum(a*b for a,b in zip(c,inner.basis_values(tuple(old),x,order)))==sum(a*b for a,b in zip(mapped,inner.basis_values(tuple(new),x,order)))


def test_retargeted_model_reuses_global_proof_and_checks_new_functional(monkeypatch):
    q=quotes();base,_=inner.baseline(q,[F(1,25)],LO,HI)
    doc=inner.search(q,W,LO,HI,base,inner.spec(W,q,6,6),quadrature_cells=16,time_limit=15,endpoints=('maximum',))[0]
    cache={};inner.verify_witness(doc,q,W,core_cache=cache)
    original_constraints=inner.constraints
    def forbidden(*a,**kw):raise AssertionError('Identical global model core was rebuilt')
    monkeypatch.setattr(inner,'constraints',forbidden)
    w=Window.make('3/8',1,'1/32','1/20')
    new=inner.retarget(doc,q,w,core_cache=cache)
    assert new['retargeted_from_window']==W.json() and new['theta']==doc['theta']
    assert inner.verify_witness(new,q,w,core_cache=cache)['verified']
    # Independent Gaussian quadrature and SciPy splines, split at every knot.
    ts=list(map(lambda x:float(F(x)),doc['definition']['time_knots']))
    ks=list(map(lambda x:float(F(x)),doc['definition']['strike_knots']))
    theta=np.asarray(list(map(lambda x:float(F(x)),doc['theta']))).reshape(len(ts)-4,len(ks)-4)
    def nodes(knots,left,right):
        breaks=sorted({left,right}|{x for x in knots if left<x<right});out=[]
        z,weights=np.polynomial.legendre.leggauss(12)
        for a,b in zip(breaks[:-1],breaks[1:]):
            out.extend(zip((a+b)/2+(b-a)*z/2,(b-a)*weights/2))
        return out
    numerator=0.;denom=0.
    for t,wt in nodes(ts,float(w.t-w.ht),float(w.t+w.ht)):
        bt=scipy_basis(ts,t);dt=scipy_basis(ts,t,1)
        for k,wk in nodes(ks,float(w.k-w.hk),float(w.k+w.hk)):
            bk=scipy_basis(ks,k);ddk=scipy_basis(ks,k,2)
            v=.04*t;d2=(-np.log(k)-v/2)/np.sqrt(v);g=k*np.exp(-d2*d2/2)/np.sqrt(2*np.pi*v)
            phi=(1-((t-float(w.t))/float(w.ht))**2)**3*(1-((k-float(w.k))/float(w.hk))**2)**3
            numerator+=wt*wk*phi*(.04*g+2*(dt@theta@bk))
            denom+=wt*wk*phi*(g+k*k*(bt@theta@ddk))
    actual=numerator/denom
    assert float(F(new['average']['lo']))-1e-12<=actual<=float(F(new['average']['hi']))+1e-12
    bad=deepcopy(new);bad['average']['lo']='0'
    with pytest.raises(ValueError,match='average claim'):inner.verify_witness(bad,q,w,core_cache=cache)
    monkeypatch.setattr(inner,'constraints',original_constraints)
    bad=deepcopy(new);bad['theta'][0]='10000'
    with pytest.raises(ValueError,match='Bernstein'):inner.verify_witness(bad,q,w,core_cache=cache)


def test_continuum_region_and_arbitrarily_narrow_ramps():
    q=quotes();region=time_models.full_prior_region(q,F(1,25),LO,HI)
    strip=next(x for x in region['strips'] if x['left']=='1/4')
    assert F(strip['strict_time_support_width_upper'])==F(5,66)
    # A fixed number of ramp halvings cannot handle a window this close to
    # the strict boundary; the algebraic choice must still attain both priors.
    ht=F(5,132)*(1-F(1,10**12));w=Window.make('1/2',1,ht,'1/10')
    docs=time_models.candidates(q,w,F(1,25),LO,HI,adaptive_ramps=True)
    assert [F(x['target_variance']) for x in docs]==[LO,HI]
    for doc in docs:assert time_models.verify(doc,q,w)['verified']


@pytest.mark.parametrize('w',[Window.make('1/8',1,'1/100','1/10'),Window.make('5/4',2,'1/10','1/4')])
def test_first_gap_and_unobserved_future_have_exact_endpoints(w):
    q=quotes();docs=time_models.candidates(q,w,F(1,25),LO,HI,adaptive_ramps=True)
    assert [F(doc['target_variance']) for doc in docs]==[LO,HI]
    for doc in docs:assert time_models.verify(doc,q,w)['verified']


def test_positive_complete_window_and_no_solver_verification(tmp_path,monkeypatch):
    path=tmp_path/'quotes.csv';write_quotes(path,quotes())
    out=tmp_path/'run';w=Window.make('3/8',1,'1/64','1/10')
    result=tightening.run(path,out,w,LO,HI,outer_levels=(4,),spline_levels=(4,),
        positive_denominator=True,denominator_levels=(),expanded_models=True,time_limit=20,progress_mode='off')
    assert result['gap']['approximate_endpoint_gaps']==[0.,0.]
    assert F(result['denominator_floor']['lower_bound'])>0
    def forbidden(*a,**kw):raise AssertionError('Optimizer restarted during proof verification')
    monkeypatch.setattr(denominator,'optimize',forbidden);monkeypatch.setattr(certlp,'optimize',forbidden)
    checked=tightening.verify(out)
    assert checked['denominator_positive_proved'] and not checked['optimizer_restarted']


def test_finite_frontier_closure_does_not_claim_all_kernels(tmp_path,monkeypatch):
    path=tmp_path/'quotes.csv';write_quotes(path,quotes());out=tmp_path/'frontier'
    result=certified_frontier.run(path,out,['3/8'],['1'],['1/64','1/32'],['1/10'],
        outer_levels=(4,),spline_levels=(4,),denominator_levels=(),time_limit=20,progress_mode='off')
    assert result['requested_grid_certified'] and not result['universal_frontier_proved']
    def forbidden(*a,**kw):raise AssertionError('Optimizer restarted during frontier verification')
    monkeypatch.setattr(denominator,'optimize',forbidden);monkeypatch.setattr(certlp,'optimize',forbidden)
    assert certified_frontier.verify(out)['requested_grid_certified']
    result['universal_frontier_proved']=True;save_json(out/'frontier.json',result)
    manifest=json.loads((out/'frontier_manifest.json').read_text(encoding='utf-8'))
    manifest['artifacts']['frontier.json']=sha256(out/'frontier.json');save_json(out/'frontier_manifest.json',manifest)
    with pytest.raises(ValueError,match='completion claim'):certified_frontier.verify(out)
