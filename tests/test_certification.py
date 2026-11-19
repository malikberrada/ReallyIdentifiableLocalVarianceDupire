from decimal import Decimal, localcontext
from fractions import Fraction as F
import json
import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import linprog
from dupire_identify.interval import I,exp_point,log_point,sqrt_point,pi,black_call
from dupire_identify.continuous import Window,outer_bounds,price_envelope
from dupire_identify.workflow import synthetic_quotes,run_identification,verify_run
from dupire_identify.witness import witness_pair
from dupire_identify.lp import Polytope,exact_dual_lower,save_certificate,verify_certificate
from dupire_identify.io import Quote
from dupire_identify.p2 import certify_polynomial
from dupire_identify.polynomial import bernstein_range,p2_range


def dec(q):return Decimal(q.numerator)/Decimal(q.denominator)


@pytest.mark.parametrize('x',[F(0),F(1,100),F(1,3),F(2),F(12),F(-3)])
def test_exp_encloses_independent_decimal(x):
    r=exp_point(x)
    with localcontext() as ctx:
        ctx.prec=90;value=dec(x).exp()
        assert dec(r.lo)<=value<=dec(r.hi)
        assert r.hi-r.lo<F(1,10**30)


@pytest.mark.parametrize('x',[F(1,100),F(1,3),F(2),F(12345,100)])
def test_log_sqrt_enclose_decimal(x):
    with localcontext() as ctx:
        ctx.prec=90
        for interval,value in [(log_point(x),dec(x).ln()),(sqrt_point(x),dec(x).sqrt())]:
            assert dec(interval.lo)<=value<=dec(interval.hi)


def test_pi_and_interval_algebra():
    with localcontext() as ctx:
        ctx.prec=90
        known=Decimal('3.141592653589793238462643383279502884197169399375105820974944592307816406286')
        assert dec(pi().lo)<known<dec(pi().hi)
    assert I(-2,3)*I(-4,5)==I(-12,15)
    r=I(F(-1,3),F(1,7)).rounded(10)
    assert r.lo<=F(-1,3) and r.hi>=F(1,7)


@pytest.mark.parametrize('k',[F(1,2),F(1),F(3,2),F(3)])
def test_black_interval_matches_numerical(k):
    from dupire_identify.backend import Backend
    price=black_call(k,F(1,25))
    numerical=Backend('numpy').black([float(k)],[.04])[0,0]
    assert abs(float((price.lo+price.hi)/2)-numerical)<3e-15
    assert price.hi-price.lo<F(1,10**29)


def test_black_deep_wings_do_not_overflow():
    for k in (F(1,1000000),F(1000000)):
        price=black_call(k,F(1,100000))
        assert price.lo>=max(F(0),1-k)
        assert price.hi>=price.lo
        assert price.hi-price.lo<F(1,10**25)


def test_exact_dual_retains_stationarity_residual():
    # min -x on [0,1] = -1. A deliberately inexact dual must not claim -0.99.
    A=sparse.csr_matrix([[1.]])
    bound=exact_dual_lower(A,[1.],[0.],[1.],[-1.],[.99])
    assert bound==F(-1)
    # Negative bounds and coefficients are also supported.
    A=sparse.csr_matrix([[-1.,2.]])
    bound=exact_dual_lower(A,[2.],[-2.,-3.],[4.,2.],[1.,-3.],[.234])
    res=linprog([1,-3],A_ub=A,b_ub=[2],bounds=[(-2,4),(-3,2)])
    assert float(bound)<=res.fun+1e-12


def test_certificate_reverification_and_tamper(tmp_path):
    A=sparse.csr_matrix([[1.,1.]])
    p=Polytope(A,np.array([1.]),np.zeros(2),np.ones(2),None,None,[])
    f=np.array([-2.,-1.]);res=linprog(f,A_ub=A,b_ub=p.b,bounds=[(0,1)]*2)
    bound,doc=save_certificate(tmp_path,'proof',p,f,res)
    assert bound==F(-2)
    assert verify_certificate(tmp_path/'proof.json')['verified']
    doc['lower_bound']='-1'
    (tmp_path/'proof.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='claimed dual'):
        verify_certificate(tmp_path/'proof.json')


def test_bernstein_bounds_polynomial_between_samples():
    # f(x)=x-x^2 peaks between endpoints.
    r=bernstein_range([F(0),F(1),F(-1)],F(0),F(1))
    assert r.lo<=0 and r.hi>=F(1,4)
    r=p2_range({(1,1):F(1),(2,0):F(-1)},F(0),F(1),F(0),F(2))
    assert r.lo<=-1 and r.hi>=1


def test_continuous_outer_contains_known_model(tmp_path):
    quotes=synthetic_quotes(tmp_path/'q.csv')
    w=Window.make('3/8',1,'1/16','1/10')
    for cells in (4,8):
        r=outer_bounds(quotes,w,'1/400','1/4',cells)
        assert not r['empty_class']
        assert F(r['lower'])<=F(1,25)<=F(r['upper'])
        assert not r['sharpness_proved']


def test_price_envelope_covers_black_model(tmp_path):
    quotes=synthetic_quotes(tmp_path/'q.csv')
    ta,tb,ka,kb=F(1,3),F(2,5),F(9,10),F(11,10)
    envelope=price_envelope(quotes,ta,tb,ka,kb,F(1,4))
    for t in (ta,(ta+tb)/2,tb):
        for k in (ka,(ka+kb)/2,kb):
            c=black_call(k,F(1,25)*t)
            assert envelope.lo<=c.lo<=c.hi<=envelope.hi


def test_continuous_witness_pair(tmp_path):
    quotes=synthetic_quotes(tmp_path/'q.csv')
    r=witness_pair(quotes)
    assert r['certified']
    assert .02828<float(F(r['separation_lower']))<.02829
    assert .01414<float(F(r['minimax_error_lower']))<.01415
    assert all(p['within_bid_ask'] for p in r['common_prices'])
    assert r['pricing_errors']['discretization']=='0'
    with pytest.raises(ValueError,match='maturity'):
        witness_pair(quotes,left='1/10',right='1/2')


def test_incompatible_quotes_never_certify_witnesses():
    quotes=[Quote('a',F(1,4),F(1),F(1,1000),F(2,1000)),
            Quote('b',F(3,4),F(1),F(1,1000),F(2,1000))]
    assert not witness_pair(quotes)['certified']


def test_p2_uniform_residual_is_valid_and_honest():
    config=dict(call_polynomial=[[0,0,'1']],amin='1/400',amax='1/4',
                kmin='1/10',kmax='2',tmax='1',cells=2)
    r=certify_polynomial(config)
    assert F(r['eta'])==1
    assert F(r['residual_sup'])==0
    assert r['scope']=='all_bounded_variance_continuous_diffusions'
    config['variance_polynomial']=[[0,0,'2']]
    with pytest.raises(ValueError,match='Variance bounds'):
        certify_polynomial(config)


def test_p2_nonconstant_polynomial_residual():
    config=dict(call_polynomial=[[0,0,'1'],[0,1,'-1'],[1,2,'1/10']],
                variance_polynomial=[[0,0,'1/25']],amin='1/25',amax='1/25',
                kmin='1/10',kmax='2',tmax='1',cells=4)
    r=certify_polynomial(config)
    assert F(r['residual_sup'])==F(2,5)
    assert F(r['initial_error'])==1
    assert r['scope']=='fixed_variance_on_domain_bounded_exterior'


def test_run_and_verification_without_solver_reentry(tmp_path,monkeypatch):
    qpath=tmp_path/'q.csv';synthetic_quotes(qpath)
    output=tmp_path/'run'
    result=run_identification(qpath,output,levels=(4,),nt=8,nk=19,
                              discrete=False,backend='numpy',progress_mode='off')
    assert result['decision']['status']=='requested_precision_impossible_by_continuous_witnesses'
    import dupire_identify.lp as lp
    monkeypatch.setattr(lp,'linprog',lambda *a,**k:(_ for _ in ()).throw(AssertionError('solver used')))
    assert verify_run(output)['verified']
    resumed=run_identification(qpath,output,levels=(4,),nt=8,nk=19,
                               discrete=False,backend='numpy',progress_mode='off',resume=True)
    assert resumed['decision']==result['decision']


def test_empty_class_report_is_verifiable(tmp_path):
    from dupire_identify.io import write_quotes
    quotes=[Quote('high',F(1,4),F(1),F(16,100),F(17,100)),
            Quote('low',F(1,4),F(1),F(1,100),F(2,100)),
            Quote('later',F(3,4),F(1),F(1,100),F(3,100))]
    path=tmp_path/'q.csv';write_quotes(path,quotes)
    out=tmp_path/'empty'
    result=run_identification(path,out,window=Window.make('1/4',1,'1/20','1/10'),
        levels=(8,),discrete=False,backend='numpy',progress_mode='off')
    assert result['status']=='empty_continuous_class'
    assert verify_run(out)['empty_class']
