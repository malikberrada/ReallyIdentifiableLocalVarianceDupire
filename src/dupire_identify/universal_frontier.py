"""Certified continuum atlas, with explicit open regions and cover checks.

A leaf denotes EVERY parameter in a closed rational box, not a sampled grid.
Only symbolic region inclusion closes a leaf. Finite budgets never imply a
theorem on the unbounded four-dimensional parameter space.
"""
from fractions import Fraction as F
from pathlib import Path
import html
import json
import shutil
from . import exact_endpoint
from .io import save_json,sha256
from .progress import Progress,QUIET

KEYS=('t','k','ht','hk')
DEFAULT_DOMAIN={'t':['3/8','3/8'],'k':['1','1'],'ht':['1/64','1/16'],'hk':['1/20','1/10']}


def parse_domain(doc):
    if set(doc)!=set(KEYS):raise ValueError('Domain keys must be t,k,ht,hk')
    box={k:tuple(map(F,doc[k])) for k in KEYS}
    if any(len(box[k])!=2 or box[k][0]>box[k][1] for k in KEYS):raise ValueError('Invalid parameter bounds')
    if min(box['ht'][0],box['hk'][0])<=0 or box['t'][0]-box['ht'][1]<=0 or box['k'][0]-box['hk'][1]<=0:
        raise ValueError('Every kernel in the domain must lie in positive time and strike')
    return box


def payload(box):return {k:list(map(str,box[k])) for k in KEYS}


def include(box,atlas):
    tl,tr=box['t'][0]-box['ht'][1],box['t'][1]+box['ht'][1]
    kl,kr=box['k'][0]-box['hk'][1],box['k'][1]+box['hk'][1]
    for i,rect in enumerate(atlas['rectangles']):
        if (F(rect['t_left'])<=tl and tr<=F(rect['t_right']) and F(rect['k_left'])<=kl and kr<=F(rect['k_right'])):
            return dict(kind='exact_plateau_rectangle',index=i,interval=rect['identification_interval'])
    temporal=atlas['temporal_strips']
    if temporal:
        for i,strip in enumerate(temporal['strips']):
            if F(strip['left'])<tl and tr<F(strip['right']) and 2*box['ht'][1]<F(strip['strict_time_support_width_upper']):
                return dict(kind='exact_temporal_strip',index=i,interval=temporal['identification_interval'])
        if tl>F(temporal['after_last_maturity']['strict_time_support_left_lower']):
            return dict(kind='exact_unobserved_future',interval=temporal['identification_interval'])
    return None


def divide(box,axis,value):
    left=dict(box);right=dict(box);left[axis]=(box[axis][0],value);right[axis]=(value,box[axis][1]);return left,right


def cover(box,atlas,max_leaves=128,max_depth=8,progress=QUIET):
    leaves=0
    def visit(b,depth,budget):
        nonlocal leaves
        proof=include(b,atlas)
        if proof is not None:
            leaves+=1;progress.event('continuum_parameter_box','certified',done=leaves,region=proof['kind'])
            return dict(box=payload(b),status='exact',proof=proof)
        widths={k:(b[k][1]-b[k][0])/max(F(1,100),abs((b[k][1]+b[k][0])/2)) for k in KEYS}
        axis=max(KEYS,key=lambda k:widths[k])
        if depth>=max_depth or budget<=1 or widths[axis]==0:
            leaves+=1;progress.event('continuum_parameter_box','open',done=leaves,depth=depth)
            return dict(box=payload(b),status='open',reason='No region theorem covers this whole box within the requested budget')
        split=sum(b[axis])/2;left,right=divide(b,axis,split)
        return dict(box=payload(b),status='split',axis=axis,split=str(split),
            children=[visit(left,depth+1,budget//2),visit(right,depth+1,budget-budget//2)])
    if max_leaves<1 or max_depth<0:raise ValueError('Invalid coverage budget')
    return visit(box,0,max_leaves)


def check_tree(node,box,atlas):
    if node['box']!=payload(box):raise ValueError('Parameter box or partition changed')
    status=node['status']
    if status=='exact':
        proof=include(box,atlas)
        if proof is None or proof!=node['proof']:raise ValueError('Unsupported universal endpoint claim')
        return 1,0
    if status=='open':return 0,1
    if status!='split' or node['axis'] not in KEYS or len(node['children'])!=2:raise ValueError('Invalid continuum cover node')
    axis=node['axis'];middle=F(node['split'])
    if not box[axis][0]<middle<box[axis][1]:raise ValueError('Empty or overlapping partition')
    left,right=divide(box,axis,middle)
    a,b=check_tree(node['children'][0],left,atlas);c,d=check_tree(node['children'][1],right,atlas)
    return a+c,b+d


def run(certificate,output,domain_file=None,max_leaves=128,max_depth=8,progress_mode='plain',heartbeat=2):
    source=Path(certificate).resolve();out=Path(output).resolve()
    if source==out or source.is_relative_to(out) or out.is_relative_to(source):
        raise ValueError('Use a separate output directory for the continuum frontier')
    if (out/'universal_manifest.json').exists():raise ValueError('Use another output directory')
    out.mkdir(parents=True,exist_ok=True);progress=Progress(progress_mode,heartbeat,out/'events.jsonl')
    with progress.phase('verify_exact_diffusion'):exact_endpoint.verify(source,progress)
    # Copy only tracked certificate artifacts so the atlas is portable.
    sm=json.loads((source/'exact_manifest.json').read_text(encoding='utf-8'));target=out/'exact_seed';target.mkdir(exist_ok=True)
    for name in list(sm['artifacts'])+['exact_manifest.json']:shutil.copy2(source/name,target/name)
    domain=json.loads(Path(domain_file).read_text(encoding='utf-8')) if domain_file else DEFAULT_DOMAIN
    box=parse_domain(domain);atlas=json.loads((target/'regions.json').read_text(encoding='utf-8'))
    tree=cover(box,atlas,max_leaves,max_depth,progress);exact,open_count=check_tree(tree,box,atlas)
    result=dict(kind='verified_continuum_frontier_v1',domain=payload(box),tree=tree,
        exact_leaf_boxes=exact,open_leaf_boxes=open_count,all_parameters_in_domain_covered=True,
        every_window_in_requested_domain_solved_exactly=open_count==0,
        unbounded_parameter_frontier_solved=False,sampled_grid_is_proof=False,
        status='entire_requested_continuum_exact' if open_count==0 else 'requested_continuum_has_open_regions',
        scope='Every window in the specified closed parameter domain; open leaves are explicitly unresolved')
    save_json(out/'frontier.json',result)
    page='<!doctype html><html lang="fr"><meta charset="utf-8"><title>Frontiere continue</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto}pre{white-space:pre-wrap}</style><h1>Frontiere sur un continuum de fenetres</h1><p>'+html.escape(result['status'])+'</p><p>Chaque feuille couvre tous les parametres de sa boite. Les regions ouvertes restent ouvertes. Aucune grille finie ne remplace cette preuve.</p><pre>'+html.escape(json.dumps(result,indent=2))+'</pre><p>La frontiere de tous les parametres positifs, sans domaine borne, reste ouverte.</p></html>'
    (out/'frontier.html').write_text(page,encoding='utf-8')
    names=['frontier.json','frontier.html','exact_seed/exact_manifest.json']+['exact_seed/'+n for n in sm['artifacts']]
    save_json(out/'universal_manifest.json',dict(status='completed',artifacts={name:sha256(out/name) for name in names}))
    return result


def verify(directory,progress=QUIET):
    out=Path(directory);m=json.loads((out/'universal_manifest.json').read_text(encoding='utf-8'))
    if m['status']!='completed':raise ValueError('Incomplete continuum certificate')
    seed=json.loads((out/'exact_seed/exact_manifest.json').read_text(encoding='utf-8'))
    names={'frontier.json','frontier.html','exact_seed/exact_manifest.json'}|{'exact_seed/'+n for n in seed['artifacts']}
    if set(m['artifacts'])!=names:raise ValueError('Missing continuum artifact')
    for name,digest in m['artifacts'].items():
        if sha256(out/name)!=digest:raise ValueError('Changed continuum artifact: '+name)
    exact_endpoint.verify(out/'exact_seed',progress)
    result=json.loads((out/'frontier.json').read_text(encoding='utf-8'));box=parse_domain(result['domain'])
    atlas=json.loads((out/'exact_seed/regions.json').read_text(encoding='utf-8'))
    exact,opened=check_tree(result['tree'],box,atlas)
    expected=dict(kind='verified_continuum_frontier_v1',domain=payload(box),tree=result['tree'],exact_leaf_boxes=exact,open_leaf_boxes=opened,
        all_parameters_in_domain_covered=True,every_window_in_requested_domain_solved_exactly=opened==0,
        unbounded_parameter_frontier_solved=False,sampled_grid_is_proof=False,
        status='entire_requested_continuum_exact' if opened==0 else 'requested_continuum_has_open_regions',
        scope='Every window in the specified closed parameter domain; open leaves are explicitly unresolved')
    if result!=expected:raise ValueError('Incorrect continuum completion claim')
    return dict(verified=True,all_parameters_in_domain_covered=True,every_window_in_requested_domain_solved_exactly=opened==0,
        exact_leaf_boxes=exact,open_leaf_boxes=opened,unbounded_parameter_frontier_solved=False,
        optimizer_restarted=False,pde_solver_restarted=False)
