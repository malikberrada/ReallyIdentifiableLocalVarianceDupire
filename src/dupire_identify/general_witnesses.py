"""Executable extremal coefficient constructions in general certified regions."""
from fractions import Fraction as F
from pathlib import Path
import json
from . import fixed_model,general_frontier,general_theorems
from .families import Family
from .io import save_json,sha256,quote_payload
from .interval import I
from .progress import QUIET


def result(base,quotes,pricing,w):
    theorem=general_theorems.regions(base,quotes,pricing);proof=general_theorems.covers(Family.point(w),theorem)
    if proof is None:raise ValueError('The selected window is outside the proved future/strict-slack regions')
    change=F(0) if proof['kind']=='unobserved_future' else F(theorem['strict_slack'])/2
    intervals=[]
    for q,p in zip(quotes,pricing['prices']):
        price=I(F(p['price']['lo'])-change,F(p['price']['hi'])+change)
        if price.lo<q.bid or price.hi>q.ask:raise ArithmeticError('Coupling bound leaves the quote spread')
        intervals.append(dict(id=q.id,price=price.json(),compatible=True))
    return dict(window=w.json(),quotes=quote_payload(quotes),proof=proof,theorem=theorem,
        minimum=I(F(base['amin'])).json(),maximum=I(F(base['amax'])).json(),
        maximum_price_change=str(change),price_intervals_for_both_models=intervals,
        denominator_positivity='strict, by nondegenerate positive-state diffusion and the positive compact bump',
        numeric_denominator_lower=None,all_positive_parameters_solved=False,all_smiles_solved=False)


def run(source,output,w,progress=QUIET):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'witness_manifest.json').exists():raise ValueError('Use a new witness output directory')
    source=Path(source).resolve()
    if source==out.resolve() or out.resolve().is_relative_to(source) or source.is_relative_to(out.resolve()):raise ValueError('Use a separate construction output')
    general_frontier.tracked_model(source,out/'base');fixed_model.verify(out/'base',progress);m=fixed_model.Model(out/'base')
    report=result(m.config,m.quotes,m.pricing,w)
    for label,target in (('minimum',F(m.config['amin'])),('maximum',F(m.config['amax']))):
        cfg=general_theorems.endpoint_config(m.config,w,target)
        if fixed_model.coefficient_range(cfg,Family.point(w))!=I(target):raise ArithmeticError('Constructed coefficient fails its plateau identity')
        save_json(out/(label+'.json'),cfg)
    save_json(out/'result.json',report)
    mname='model_manifest.json' if (out/'base/model_manifest.json').exists() else 'exact_manifest.json'
    manifest=json.loads((out/'base'/mname).read_text(encoding='utf-8'))
    names=['minimum.json','maximum.json','result.json','base/'+mname]+['base/'+n for n in manifest['artifacts']]
    save_json(out/'witness_manifest.json',dict(status='completed',artifacts={n:sha256(out/n) for n in names}))
    return report


def verify(directory,progress=QUIET):
    from .continuous import Window
    out=Path(directory);manifest=json.loads((out/'witness_manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed':raise ValueError('Incomplete witness construction')
    for n,h in manifest['artifacts'].items():
        p=Path(n)
        if p.is_absolute() or '..' in p.parts or sha256(out/p)!=h:raise ValueError('Changed general witness artifact')
    fixed_model.verify(out/'base',progress);model=fixed_model.Model(out/'base')
    saved=json.loads((out/'result.json').read_text(encoding='utf-8'));w=Window.make(**saved['window'])
    if saved!=result(model.config,model.quotes,model.pricing,w):raise ValueError('Incorrect coupling construction claim')
    for label,target in (('minimum',F(model.config['amin'])),('maximum',F(model.config['amax']))):
        cfg=json.loads((out/(label+'.json')).read_text(encoding='utf-8'))
        if cfg!=general_theorems.endpoint_config(model.config,w,target) or fixed_model.coefficient_range(cfg,Family.point(w))!=I(target):
            raise ValueError('Modified extremal diffusion')
    return dict(verified=True,minimum=saved['minimum'],maximum=saved['maximum'],
        optimizer_restarted=False,pde_solver_restarted=False,all_positive_parameters_solved=False,all_smiles_solved=False)
