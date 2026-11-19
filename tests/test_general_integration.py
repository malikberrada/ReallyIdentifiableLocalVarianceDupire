"""End-to-end proof checks, including semantic tampering after rehashing."""
from fractions import Fraction as F
import json
import pytest
from dupire_identify import fixed_model,general_frontier,general_witnesses,exhaustion
from dupire_identify.families import Family
from dupire_identify.continuous import Window
from dupire_identify.interval import black_call
from dupire_identify.io import Quote,write_quotes,save_json,sha256


@pytest.fixture(scope='module')
def certified_model(tmp_path_factory):
    root=tmp_path_factory.mktemp('general_model');csv=root/'q.csv';q=[]
    for t in (F(3,10),F(7,10)):
        for k in (F(4,5),F(1),F(6,5)):
            p=black_call(k,t/F(25))
            q.append(Quote(str(t)+'_'+str(k),t,k,max(F(0),p.lo-F(1,500)),p.hi+F(1,500)))
    write_quotes(csv,q)
    config=dict(kind='piecewise_quintic_plateau_diffusion_v1',amin='1/400',amax='1/4',
        prefix_end='3/20',prefix_variance='1/25',strike_knots=['1/8','13/20','7/5','8'],
        price_envelope_intersection=True,
        phases=[dict(left='3/20',right='7/10',variance=['1/25']*4)])
    model=root/'model'
    proof=fixed_model.run(config,csv,model,spatial_cells=32,pulse_cells=4,time_steps=64,
        refinements=1,storage_stride=8,progress_mode='off')
    assert proof['all_quotes_compatible']
    return csv,model


def rehash(directory,manifest,artifact):
    path=directory/manifest;doc=json.loads(path.read_text())
    doc['artifacts'][artifact]=sha256(directory/artifact);save_json(path,doc)


def test_future_continuum_resume_and_semantic_universal_forgery(certified_model,tmp_path):
    csv,model=certified_model
    family=Family.make(dict(coordinates='left',u=['1','2'],v=['1/100','10'],ht=['1/100','2'],hk=['1/100','3']))
    out=tmp_path/'frontier'
    options=dict(model_runs=[model],fit_models=0,max_leaves=1,max_depth=0,progress_mode='off')
    result=general_frontier.run(csv,out,family,**options)
    assert result['all_parameters_in_requested_domain_solved'] and result['model_class_nonempty_certified']
    assert result['tree']['analytic']['theorem']['kind']=='unobserved_future'
    assert general_frontier.verify(out)['verified']
    assert general_frontier.run(csv,out,family,resume=True,**options)==result
    result['all_positive_parameters_solved']=True;save_json(out/'frontier.json',result)
    rehash(out,'general_manifest.json','frontier.json')
    with pytest.raises(ValueError,match='completion'):general_frontier.verify(out)


def test_data_guided_partition_covers_nonmidpoint_cut_and_keeps_open_leaf(certified_model,tmp_path):
    csv,model=certified_model
    family=Family.make(dict(coordinates='left',u=['3/5','1'],v=['9/10','9/10'],ht=['1/20','1/20'],hk=['1/10','1/10']))
    out=tmp_path/'partition'
    answer=general_frontier.run(csv,out,family,model_runs=[model],fit_models=0,max_leaves=2,max_depth=1,
        strike_cells=2,target_cells=2,outer_steps=0,lp_seconds=5,progress_mode='off')
    assert answer['tree']['status']=='split' and F(answer['tree']['middle'])==F(7,10)
    assert answer['tree']['partition_reason']=='last_quote_maturity'
    assert answer['tree']['children'][1]['status']=='exact'
    assert answer['open_leaf_boxes']==1 and not answer['all_parameters_in_requested_domain_solved']
    assert general_frontier.verify(out)['verified']


def test_no_inner_model_never_turns_lp_feasibility_into_existence(certified_model,tmp_path):
    csv,model=certified_model
    family=Family.point(Window.make('1/2',1,'1/20','1/10'))
    answer=general_frontier.run(csv,tmp_path/'outer_only',family,fit_models=0,max_leaves=1,max_depth=0,
        strike_cells=2,target_cells=2,outer_steps=0,lp_seconds=5,progress_mode='off')
    assert not answer['model_class_nonempty_certified']
    assert not answer['all_parameters_in_requested_domain_solved']
    assert answer['tree']['gaps']['status']=='outer_only_no_verified_model'
    assert general_frontier.verify(tmp_path/'outer_only')['verified']


def test_empty_quote_class_has_no_identification_endpoints(tmp_path):
    csv=tmp_path/'bad.csv'
    write_quotes(csv,[Quote('low',F(1),F(9,10),F(1,10),F(11,100)),
        Quote('high',F(1),F(11,10),F(1,5),F(21,100))])
    family=Family.point(Window.make('1/2',1,'1/20','1/10'))
    result=general_frontier.run(csv,tmp_path/'empty',family,fit_models=0,progress_mode='off')
    assert result['status']=='empty_model_class' and result['model_class_empty_certified']
    assert 'lower' not in result['tree']
    assert general_frontier.verify(tmp_path/'empty')['verified']


def test_finite_exhaustion_cannot_claim_infinite_completion(certified_model,tmp_path):
    csv,model=certified_model;out=tmp_path/'shells'
    answer=exhaustion.run(csv,out,indices=(0,),model_runs=[model],fit_models=0,
        amin='.0025',amax='.25',tolerance='.01',max_leaves=1,max_depth=0,progress_mode='off')
    assert answer['all_requested_shells_solved']
    assert answer['mathematical_scheme_covers_all_positive_parameters']
    assert not answer['all_positive_parameters_solved'] and answer['finite_prefix_only']
    assert exhaustion.verify(out)['verified']
    answer['all_positive_parameters_solved']=True;save_json(out/'exhaustion.json',answer)
    rehash(out,'exhaustion_manifest.json','exhaustion.json')
    with pytest.raises(ValueError,match='completion'):exhaustion.verify(out)


def test_general_witnesses_are_actual_coefficients_and_reject_rehashed_changes(certified_model,tmp_path):
    csv,model=certified_model;out=tmp_path/'witnesses'
    proof=general_witnesses.run(model,out,Window.make(2,1,'1/5','1/4'))
    assert proof['maximum_price_change']=='0'
    assert general_witnesses.verify(out)['verified']
    altered=json.loads((out/'maximum.json').read_text());altered['prefix_variance']='1/20'
    save_json(out/'maximum.json',altered);rehash(out,'witness_manifest.json','maximum.json')
    with pytest.raises(ValueError,match='extremal'):general_witnesses.verify(out)
