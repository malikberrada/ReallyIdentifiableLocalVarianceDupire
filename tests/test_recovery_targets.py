import json
from fractions import Fraction as F
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from dupire_identify import certlp,clock_models,fixed_model,calibrate_general
from dupire_identify.lp import Polytope
from dupire_identify.display import approximate
from dupire_identify.interval import I,black_call
from dupire_identify.families import Family
from dupire_identify.continuous import Window
from dupire_identify.io import Quote,write_quotes,save_json,sha256
from dupire_identify.progress import Progress


def flat_quotes():
    quotes=[]
    for t in (F(1,4),F(3,4),F(1)):
        for k in (F(4,5),F(1),F(6,5)):
            p=black_call(k,F(1,25)*t)
            quotes.append(Quote(f'{t}_{k}',t,k,max(F(0),p.lo-F(1,1000)),p.hi+F(1,1000)))
    return quotes


def target():
    return Family.make(dict(t=['39/64','41/64'],k=['19/20','21/20'],ht=['1/128','1/64'],hk=['1/40','1/10']))


def test_overflow_preserves_exact_certificate_and_strict_json(tmp_path):
    # Finite binary LP data, but their exact dual product exceeds binary64.
    p=Polytope(sparse.csr_matrix([[1.]]),np.array([1e300]),np.array([0.]),np.array([1.]),None,None,[])
    res=SimpleNamespace(ineqlin=SimpleNamespace(marginals=np.array([-1e100])))
    value=certlp.certificate(tmp_path/'dual.json',p,[F(0)],res)
    assert value < -10**399
    doc=json.loads((tmp_path/'dual.json').read_text())
    assert doc['approximate'] is None and F(doc['lower_bound'])==value
    assert certlp.verify(tmp_path/'dual.json')[2]==value
    assert approximate(value) is None
    save_json(tmp_path/'interval.json',I(value,0).json())
    assert json.loads((tmp_path/'interval.json').read_text())['approximate']==[None,0.]


def test_progress_does_not_convert_enormous_bounds(tmp_path):
    p=Progress('json',log=tmp_path/'events.jsonl')
    value=-F(10**600,7)
    p.event('signed_fractional_search','running',bound=approximate(value))
    assert json.loads((tmp_path/'events.jsonl').read_text())['bound'] is None


@pytest.mark.parametrize('endpoint,target_value',[('minimum',F(1,400)),('maximum',F(1,4))])
def test_new_clock_prices_and_entire_family_plateau(tmp_path,endpoint,target_value):
    q=flat_quotes();family=target();path=tmp_path/'quotes.csv';write_quotes(path,q)
    cfg=clock_models.construct(q,family,F(1,400),F(1,4),endpoint)
    assert cfg is not None
    assert fixed_model.coefficient_range(cfg,family)==I(target_value)
    proof=clock_models.run(cfg,path,tmp_path/'model')
    assert proof['all_quotes_compatible']
    assert fixed_model.verify(tmp_path/'model')['compatible']
    model=fixed_model.Model(tmp_path/'model')
    assert model.family_average(family)['average']==I(target_value).json()
    assert all(F(p['eta'])==0 for p in proof['prices'])


def test_clock_rejects_unattainable_long_plateau():
    q=flat_quotes();f=Family.point(Window.make('1/2',1,'1/5','1/10'))
    assert clock_models.construct(q,f,F(1,400),F(1,4),'maximum') is None


def test_clock_verifier_rejects_forged_price_with_rehashed_manifest(tmp_path):
    q=flat_quotes();path=tmp_path/'quotes.csv';write_quotes(path,q)
    cfg=clock_models.construct(q,target(),F(1,400),F(1,4),'minimum')
    out=tmp_path/'model';clock_models.run(cfg,path,out)
    doc=json.loads((out/'pricing.json').read_text());doc['prices'][0]['price']['hi']='0'
    save_json(out/'pricing.json',doc)
    m=json.loads((out/'model_manifest.json').read_text());m['artifacts']['pricing.json']=sha256(out/'pricing.json');save_json(out/'model_manifest.json',m)
    with pytest.raises(ValueError,match='Incorrect analytic clock prices'):fixed_model.verify(out)


def test_locked_plateau_cannot_move_during_parameter_fitting():
    q=flat_quotes();family=target()
    cfg=calibrate_general.target_template(q,family,F(1,400),F(1,4),'maximum',4)
    p=calibrate_general.Proposal(cfg,q,family.center(),nodes=40,steps=32,max_evaluations=100,backend='numpy')
    for theta in (p.initial(),np.full(p.n,.01),np.full(p.n,.2)):
        candidate=p.to_config(theta)
        assert fixed_model.coefficient_range(candidate,family)==I(F(1,4))


def test_completed_dual_is_reused_without_solver(tmp_path,monkeypatch):
    from dupire_identify import signed_outer
    p=Polytope(sparse.csr_matrix([[1.]]),np.array([.75]),np.array([0.]),np.array([1.]),None,None,[])
    f=[F(-1)];res=SimpleNamespace(ineqlin=SimpleNamespace(marginals=np.array([-1.])))
    expected=certlp.certificate(tmp_path/'c.json',p,f,res)
    assert signed_outer.saved_certificate(tmp_path/'c.json',p,f,Progress('off'))==expected
    with pytest.raises(ValueError,match='another physical relaxation'):
        signed_outer.saved_certificate(tmp_path/'c.json',p,[F(1)],Progress('off'))


def test_targeted_domains_verify_and_reject_rehashed_global_claim(tmp_path):
    from dupire_identify import targeted
    q=flat_quotes();quotes=tmp_path/'q.csv';write_quotes(quotes,q)
    domains=tmp_path/'domains.json';save_json(domains,dict(targets=[dict(name='gap',family=target().json())]))
    out=tmp_path/'targets'
    result=targeted.run(quotes,domains,out,backend='numpy',progress_mode='off')
    assert result['all_requested_targets_solved']
    assert targeted.verify(out)['all_requested_targets_solved']
    doc=json.loads((out/'targets.json').read_text());doc['all_positive_parameters_solved']=True
    save_json(out/'targets.json',doc)
    manifest=json.loads((out/'target_manifest.json').read_text())
    manifest['artifacts']['targets.json']=sha256(out/'targets.json');save_json(out/'target_manifest.json',manifest)
    with pytest.raises(ValueError,match='False target or universal completion claim'):targeted.verify(out)


def test_recovery_copies_and_reverifies_whole_prefix(tmp_path):
    from dupire_identify import exhaustion,recovery
    q=flat_quotes();quotes=tmp_path/'q.csv';write_quotes(quotes,q)
    config=clock_models.construct(q,target(),F(1,400),F(1,4),'minimum')
    clock=tmp_path/'clock';clock_models.run(config,quotes,clock)
    old=tmp_path/'old';exhaustion.run(quotes,old,indices=(0,),model_runs=[clock],fit_models=0,progress_mode='off')
    before=sha256(old/'exhaustion_manifest.json')
    new=tmp_path/'new';answer=recovery.run(old,new,progress_mode='off',backend='numpy')
    assert answer['all_requested_shells_solved']
    assert exhaustion.verify(new)['verified']
    assert sha256(old/'exhaustion_manifest.json')==before


def test_search_cache_checks_actual_bytes_and_public_verifier_stays_independent(tmp_path,monkeypatch):
    import shutil
    q=flat_quotes();quotes=tmp_path/'q.csv';write_quotes(quotes,q)
    cfg=clock_models.construct(q,target(),F(1,400),F(1,4),'minimum')
    model=tmp_path/'model';clock_models.run(cfg,quotes,model)
    fixed_model._VERIFIED_MODELS.clear()
    real=fixed_model.verify;calls=[]
    def verify(*args,**kwargs):calls.append(1);return real(*args,**kwargs)
    monkeypatch.setattr(fixed_model,'verify',verify)
    fixed_model.verify_cached(model)
    copy=tmp_path/'copy';shutil.copytree(model,copy);fixed_model.verify_cached(copy)
    assert len(calls)==1
    fixed_model.verify(copy)
    assert len(calls)==2
    doc=json.loads((copy/'pricing.json').read_text());doc['prices'][0]['price']['lo']='1'
    save_json(copy/'pricing.json',doc)
    with pytest.raises(ValueError,match='Changed fixed-model artifact'):fixed_model.verify_cached(copy)
