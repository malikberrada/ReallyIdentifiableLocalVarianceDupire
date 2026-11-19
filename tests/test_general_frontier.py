from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import json
import numpy as np
import pytest
from dupire_identify import families,general_theorems,calibrate_general,signed_outer,certlp,fixed_model
from dupire_identify.families import Family
from dupire_identify.continuous import Window
from dupire_identify.interval import I,black_call
from dupire_identify.io import Quote
from dupire_identify.native_solver import optimize


def qflat():
    q=[]
    for t in (F(3,10),F(7,10)):
        for k in (F(4,5),F(1),F(6,5)):
            p=black_call(k,F(1,25)*t)
            q.append(Quote(str(t)+'_'+str(k),t,k,max(F(0),p.lo-F(1,500)),p.hi+F(1,500)))
    return q


def exact_bump(t,k,w):
    return max(F(0),1-((t-w.t)/w.ht)**2)**3*max(F(0),1-((k-w.k)/w.hk)**2)**3


@pytest.mark.parametrize('coordinates',['center','left'])
def test_family_encloses_each_kernel_and_common_dominance(coordinates):
    d=dict(t=['7/20','2/5'],k=['49/50','51/50'],ht=['1/20','3/50'],hk=['1/10','3/25'])
    if coordinates=='left':d=dict(coordinates='left',u=['3/10','7/20'],v=['22/25','23/25'],ht=['1/20','3/50'],hk=['1/10','3/25'])
    f=Family.make(d);ref=f.common_kernel();ta,tb,ka,kb=f.support()
    import itertools
    for vals in itertools.product(*[p for p in f.intervals]):
        vals=list(vals)
        if coordinates=='left':vals[0]+=vals[2];vals[1]+=vals[3]
        w=Window.make(*vals)
        for i in range(9):
            t=ta+(tb-ta)*F(i,8)
            for j in range(9):
                k=ka+(kb-ka)*F(j,8);r=f.weights(t,t,k,k);actual=exact_bump(t,k,w)
                assert r.lo<=actual<=r.hi
                if ref:assert actual>=ref[1]*exact_bump(t,k,ref[0])


def test_left_coordinate_exhaustion_preserves_all_positive_supports():
    f=families.shell(3)
    assert f.support()[0]==f.support()[2]==F(1,8)
    assert f.cover_window().t-f.cover_window().ht>0
    for j in (0,1,4):assert families.shell(j).coordinates=='left'


def test_symbolic_huge_power_comparison_avoids_allocating_astronomical_values():
    assert not general_theorems.power_compare(F(1,100),F(1,10),10**12)
    assert general_theorems.power_compare(F(1,128),F(1,2),6)
    assert not general_theorems.power_compare(F(1,128),F(1,2),7)


def test_quick_infeasibility_is_a_sufficient_contradiction_only():
    q=qflat();assert general_theorems.quick_infeasibility(q,F(1,400),F(1,4)) is None
    bad=[Quote('a',F(1),F(9,10),F(1,10),F(11,100)),Quote('b',F(1),F(11,10),F(1,5),F(21,100))]
    assert general_theorems.quick_infeasibility(bad,F(1,400),F(1,4))['kind']=='call_monotonicity'
    bad=[Quote('a',F(1),F(4,5),F(1,5),F(21,100)),Quote('b',F(1),F(1),F(1,5),F(21,100)),Quote('c',F(1),F(6,5),F(0),F(1,100))]
    assert general_theorems.quick_infeasibility(bad,F(1,400),F(1,4))['kind']=='butterfly_convexity'


def test_phase_I_certificate_proves_empty_polytope(tmp_path):
    b=certlp.RationalLP();b.var(1);b.row({0:1},F(1,5));b.row({0:-1},-F(4,5))
    p=b.binary();phase,f=signed_outer.phase_one(p);pp=phase.binary()
    r=optimize(list(map(float,f)),pp.A,pp.b,pp.lo,pp.hi,10)
    value=certlp.certificate(tmp_path/'phase.json',pp,f,r)
    assert value>0
    assert certlp.verify(tmp_path/'phase.json')[2]==value


def test_batched_fd_matches_independent_python_and_known_black():
    q=qflat();family=Family.point(Window.make('1/2',1,'1/20','1/10'))
    cfg=calibrate_general.template(q,family,F(1,400),F(1,4),4)
    p=calibrate_general.Proposal(cfg,q,family.center(),nodes=100,steps=128,threads=2)
    z=np.stack([p.initial(),p.initial()*1.2]);z[:,0]=.04;coeff=np.einsum('bpc,nc->bpn',z[:,1:].reshape(2,p.P,p.ninside),p.basis)
    init=calibrate_general.black(p.grid[None,:],z[:,0,None]*float(F(cfg['prefix_end'])));init[:,0]=1-p.grid[0];init[:,-1]=0
    window=np.array([float(x) for x in (family.center().t,family.center().k,family.center().ht,family.center().hk)])
    ref=calibrate_general.fd_batch_python(p.grid,coeff,p.lefts,p.rights,init,128,window)
    if calibrate_general._native_batch is not None:
        actual=calibrate_general._native_batch(p.grid,coeff,p.lefts,p.rights,init,128,window,2)
        for a,b in zip(actual,ref):assert np.max(abs(a-b))<2e-12
    prices,avg=p.evaluate(z)
    assert abs(avg[0]-.04)<2e-13
    for quote,value in zip(q,prices[0]):assert abs(value-float(black_call(quote.k,F(1,25)*quote.t).lo))<.001


def test_generic_calibration_really_has_no_analytic_baseline_constraint():
    q=qflat();f=Family.point(Window.make('1/2',1,'1/20','1/10'))
    cfg=calibrate_general.template(q,f,F(1,400),F(1,4),4)
    p=calibrate_general.Proposal(cfg,q,f.center(),nodes=120,steps=128,threads=2,max_evaluations=200)
    config,diag=p.fit()
    assert diag['numerical_spread_violation']<0 and not diag['certified']
    assert config['kind']=='piecewise_quintic_plateau_diffusion_v1'
    assert 'baseline' not in config


def test_weak_functional_moments_exact_polynomial_identity():
    from dupire_identify.polynomial import integrate,mul
    poly=(F(1),F(-3),F(2))
    values=fixed_model.moments(poly,F(1,5),F(7,5),F(1,3),F(4,3))
    # Sum of cubic Bernstein basis is one on the complete cell.
    assert sum(values)==integrate(list(poly),F(1,3),F(4,3))


def test_strict_slack_region_and_future_do_not_require_a_black_smile():
    q=qflat();f=Family.point(Window.make('1/2',1,'1/20','1/10'))
    cfg=calibrate_general.template(q,f,F(1,400),F(1,4),4)
    # Use verified analytical prices here; no numerical fit flag is a proof.
    proof=dict(all_quotes_compatible=True,prices=[dict(id=z.id,price=black_call(z.k,F(1,25)*z.t).json()) for z in q])
    theorem=general_theorems.regions(cfg,q,proof)
    assert F(theorem['strict_slack'])>0
    future=Family.make(dict(coordinates='left',u=['1','2'],v=['1/100','100'],ht=['1/100','3'],hk=['1/100','10']))
    assert general_theorems.covers(future,theorem)['kind']=='unobserved_future'
    width=theorem['small_time_width'];n=-width['binary_exponent']
    h=F(width['coefficient'])/F(2**(n+2))
    tiny=Family.point(Window.make(F(1,2),1,h,F(1,10)))
    assert general_theorems.covers(tiny,theorem)['kind']=='strict_slack_small_time'


def test_constructed_coefficients_really_attain_both_prior_endpoints():
    q=qflat();family=Family.point(Window.make('1/2',1,'1/20','1/10'))
    cfg=calibrate_general.template(q,family,F(1,400),F(1,4),4)
    for w in (Window.make(2,1,'1/5','1/2'),Window.make('1/20',1,'1/100000000','1/10')):
        for target in (F(1,400),F(1,4)):
            new=general_theorems.endpoint_config(cfg,w,target)
            assert fixed_model.coefficient_range(new,Family.point(w))==I(target)
            assert F(new['prefix_end'])<w.t-w.ht


def test_missing_strict_slack_does_not_prove_a_small_window_region():
    q=qflat();f=Family.point(Window.make('1/2',1,'1/20','1/10'));cfg=calibrate_general.template(q,f,F(1,400),F(1,4),4)
    proof=dict(all_quotes_compatible=True,prices=[dict(id=z.id,price=I(z.bid,z.ask).json()) for z in q])
    doc=general_theorems.regions(cfg,q,proof)
    assert doc['small_time_width'] is None
    assert general_theorems.covers(f,doc) is None


def test_signed_family_bound_uses_physical_projection_and_recomputes(tmp_path):
    q=qflat();f=Family.make(dict(t=['49/100','51/100'],k=['99/100','101/100'],ht=['1/20','3/50'],hk=['1/10','11/100']))
    result=signed_outer.run(q,f,F(1,400),F(1,4),tmp_path,strike_cells=4,target_cells=4,time_cells=2,
        steps=2,time_limit=10,denominator_levels=(),comparison_cuts=False)
    assert not result['empty_class_certified']
    assert F(result['lower'])<=F(1,25)<=F(result['upper'])
    assert signed_outer.verify(q,f,F(1,400),F(1,4),tmp_path)==result
    bad=json.loads((tmp_path/'outer.json').read_text());bad['upper']='1/100'
    (tmp_path/'outer.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError):signed_outer.verify(q,f,F(1,400),F(1,4),tmp_path)


def test_weak_observable_contains_independent_black_average_and_density(tmp_path):
    from dupire_identify.io import write_quotes
    from dupire_identify.density import density_integral
    cfg=dict(kind='piecewise_quintic_plateau_diffusion_v1',amin='1/400',amax='1/4',prefix_end='3/20',prefix_variance='1/25',
        strike_knots=['1/8','13/20','7/5','8'],price_envelope_intersection=True,
        phases=[dict(left='3/20',right='7/10',variance=['1/25']*4)])
    q=[Quote(z.id,z.t,z.k,F(0),F(1)) for z in qflat()];csv=tmp_path/'q.csv';write_quotes(csv,q)
    p=fixed_model.run(cfg,csv,tmp_path/'model',spatial_cells=64,pulse_cells=4,time_steps=256,refinements=1,progress_mode='off')
    assert p['all_quotes_compatible'];assert fixed_model.verify(tmp_path/'model')['compatible']
    model=fixed_model.Model(tmp_path/'model');w=Window.make('1/2',1,'1/20','1/10')
    answer=model.point_average(w,force_weak=True)
    assert F(answer['average']['lo'])<=F(1,25)<=F(answer['average']['hi'])
    actual=density_integral(w,F(1,25),16)
    assert F(answer['denominator']['lo'])<=actual.lo<=actual.hi<=F(answer['denominator']['hi'])
    assert F(answer['numerator']['lo'])<=F(1,25)*actual.lo<=F(1,25)*actual.hi<=F(answer['numerator']['hi'])
