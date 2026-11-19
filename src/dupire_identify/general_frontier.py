"""Quote-driven posterior frontier on complete parameter families.

Open leaves remain open. A finite run never asserts all-smile or unbounded
parameter completion. Mathematical exhaustion and computational completion
are separate fields and separately verified.
"""
from fractions import Fraction as F
from pathlib import Path
import html
import json
import shutil
import time
from . import __version__,fixed_model,general_theorems,signed_outer,calibrate_general,adaptive_outer
from .families import Family,shell
from .interval import I
from .curves import bernstein
from .io import read_quotes,write_quotes,quote_payload,save_json,sha256
from .progress import Progress,QUIET
from .continuous import Window


def tracked_model(source,target):
    source=Path(source);target=Path(target);target.mkdir(parents=True,exist_ok=True)
    name='model_manifest.json' if (source/'model_manifest.json').exists() else 'exact_manifest.json'
    m=json.loads((source/name).read_text(encoding='utf-8'))
    for f in list(m['artifacts'])+[name]:
        path=Path(f)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Invalid model artifact path')
        dest=target/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/path,dest)


def time_clock_range(folder,family):
    path=Path(folder)/'minimum_model.json'
    if not path.exists():return None
    doc=json.loads(path.read_text(encoding='utf-8'))
    if not doc or doc.get('model_type')!='deterministic_time_change':return None
    ta,tb,ka,kb=family.support();lo=[];hi=[];end=F(doc['segments'][-1][1])
    if tb>end:lo.append(F(doc['a0']));hi.append(F(doc['a0']))
    for left,right,coeff in doc['segments']:
        left,right=F(left),F(right);a,b=max(left,ta),min(right,tb)
        if a>=b:continue
        coefficients=bernstein(list(map(F,coeff)),(a-left)/(right-left),(b-left)/(right-left),len(coeff)-1)
        lo.append(min(coefficients));hi.append(max(coefficients))
    return I(min(lo),max(hi)).json() if lo else None


def load_bank(out,quotes,amin,amax,progress,use_cache=False):
    bank=[];entries=[]
    index=json.loads((out/'bank/index.json').read_text(encoding='utf-8'))
    if len(set(e['directory'] for e in index))!=len(index):raise ValueError('Duplicate bank entry')
    for i,e in enumerate(index):
        if e['directory']!=f'bank/model_{i:03d}':raise ValueError('Invalid model bank directory')
        folder=out/e['directory']
        (fixed_model.verify_cached if use_cache else fixed_model.verify)(folder,progress);model=fixed_model.Model(folder)
        if quote_payload(model.quotes)!=quote_payload(quotes) or F(model.config['amin'])!=amin or F(model.config['amax'])!=amax:
            raise ValueError('Model bank belongs to a different quote or variance class')
        theorem=general_theorems.regions(model.config,quotes,model.pricing)
        expected=dict(directory=e['directory'],theorems=theorem)
        if expected!=e:raise ValueError('Incorrect quote-driven general theorem')
        bank.append(model);entries.append(expected)
    return bank,entries


def model_bounds(bank,family):
    records=[]
    for i,model in enumerate(bank):
        records.append(dict(model=i,kind='fixed_diffusion',bound=model.family_average(family)))
        clock=time_clock_range(model.folder,family)
        if clock is not None:records.append(dict(model=i,kind='verified_time_clock',bound=dict(method='coefficient_range',average=clock)))
    return records


def gaps(lower,upper,records,tolerance):
    if not records:return dict(status='outer_only_no_verified_model',closed=False,lower_gap=None,upper_gap=None)
    minimum=min(F(r['bound']['average']['hi']) for r in records)
    maximum=max(F(r['bound']['average']['lo']) for r in records)
    a,b=minimum-lower,upper-maximum
    if min(a,b)<0:raise ArithmeticError('Verified inner diffusion contradicts the outer continuous bound')
    return dict(status='endpoints_within_tolerance' if max(a,b)<=tolerance else 'open_endpoint_gaps',
        closed=max(a,b)<=tolerance,lower_gap=str(a),upper_gap=str(b),
        inner_minimum_upper=str(minimum),inner_maximum_lower=str(maximum))


def analytic_proof(family,bank,entries,amin,amax):
    for i,e in enumerate(entries):
        p=general_theorems.covers(family,e['theorems'])
        if p:return dict(source='quote_driven_theorem',model=i,theorem=p)
    # This is derived from the actual verified coefficients, never a
    # hard-coded list of target rectangles or a presumed Black smile.
    records=model_bounds(bank,family)
    if records and min(F(r['bound']['average']['hi']) for r in records)==amin and max(F(r['bound']['average']['lo']) for r in records)==amax:
        return dict(source='verified_models_attain_full_prior',models=records)
    return None


def final_status(tree,bank,contradiction):
    def counts(node):
        if node['status']=='split':
            values=[counts(c) for c in node['children']];return sum(v[0] for v in values),sum(v[1] for v in values)
        return (1,0) if node['status'] in ('exact','certified') else (0,1)
    a,b=counts(tree)
    return dict(certified_leaf_boxes=a,open_leaf_boxes=b,verified_model_count=len(bank),
        model_class_empty_certified=contradiction is not None,
        model_class_nonempty_certified=bool(bank),
        all_parameters_in_requested_domain_solved=bool(bank) and b==0 and contradiction is None,
        status='empty_model_class' if contradiction is not None else 'requested_domain_certified' if bank and b==0 else 'requested_domain_has_open_boxes',
        all_positive_parameters_solved=False,all_smiles_solved=False,universal_convergence_proved=False,
        scope='All windows in each certified family; one supplied finite European quote set',
        no_black_mixture_baseline_required=True)


def run(quotes_file,output,family,amin='1/400',amax='1/4',model_runs=(),fit_models=1,
        tolerance='1/100',max_leaves=8,max_depth=3,strike_cells=6,target_cells=6,outer_steps=6,
        lp_seconds=30,fit_evaluations=1500,fit_seconds=180,threads=1,
        progress_mode='plain',heartbeat=2,resume=False,mesh_refinements=1,backend='auto',retry_unresolved=False):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    q=read_quotes(quotes_file);amin,amax,tolerance=map(F,(amin,amax,tolerance))
    if not 0<amin<amax or tolerance<=0 or max_leaves<1 or max_depth<0 or fit_models<0:raise ValueError('Invalid general frontier settings')
    settings=dict(version=__version__,quotes=quote_payload(q),family=family.json(),amin=str(amin),amax=str(amax),tolerance=str(tolerance),
        max_leaves=int(max_leaves),max_depth=int(max_depth),strike_cells=int(strike_cells),target_cells=int(target_cells),outer_steps=int(outer_steps),
        lp_seconds=float(lp_seconds),fit_models=int(fit_models),fit_evaluations=int(fit_evaluations),fit_seconds=float(fit_seconds),threads=int(threads),mesh_refinements=int(mesh_refinements),
        backend=backend,retry_unresolved=bool(retry_unresolved))
    progress=Progress(progress_mode,heartbeat,out/'events.jsonl');manifest=out/'general_manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text(encoding='utf-8'))
        if not resume or old['settings']!=settings:raise ValueError('Use --resume with identical settings or another output directory')
        if old['status']=='completed':verify(out,progress);return json.loads((out/'frontier.json').read_text(encoding='utf-8'))
    save_json(manifest,dict(status='running',settings=settings));write_quotes(out/'normalized.csv',q)
    with progress.phase('quote_class_preflight'):contradiction=general_theorems.quick_infeasibility(q,amin,amax)
    entries=[];bank=[]
    if contradiction is None:
        index_path=out/'bank/index.json'
        if not index_path.exists():
            (out/'bank').mkdir(exist_ok=True)
            for source in model_runs:
                source=Path(source).resolve()
                if source==out or out.is_relative_to(source):raise ValueError('Model source must be separate from the output')
                target=out/'bank'/f'model_{len(entries):03d}';tracked_model(source,target)
                fixed_model.verify_cached(target,progress);model=fixed_model.Model(target)
                if quote_payload(model.quotes)!=quote_payload(q) or F(model.config['amin'])!=amin or F(model.config['amax'])!=amax:
                    raise ValueError('Imported model uses different observations or priors')
                entries.append(dict(directory=target.relative_to(out).as_posix(),theorems=general_theorems.regions(model.config,q,model.pricing)))
            if fit_models:
                imported=bool(entries)
                for attempt in range(fit_models):
                    endpoint=None if not imported and attempt==0 else 'minimum' if (attempt+(1 if imported else 0))%2 else 'maximum'
                    search_family=family
                    if family.support()[1]>max(z.t for z in q) or family.support()[2]<F(1,20) or family.support()[3]>5:
                        scale=sorted(z.k for z in q)[len(q)//2];first=min(z.t for z in q)
                        search_family=Family.point(Window.make(first/2,scale,first/4,scale/10))
                    candidate=out/'proposals'/f'attempt_{attempt:03d}';candidate.mkdir(parents=True,exist_ok=True)
                    try:
                        if resume and (candidate/'model_manifest.json').exists():
                            saved=json.loads((candidate/'model_manifest.json').read_text(encoding='utf-8'))
                            if not saved['compatible']:
                                progress.event('general_model_candidate','reused_unresolved',attempt=attempt+1)
                                continue
                            fixed_model.verify_cached(candidate,progress)
                            config=json.loads((candidate/'diffusion.json').read_text(encoding='utf-8'))
                            proof=json.loads((candidate/'pricing.json').read_text(encoding='utf-8'))
                        else:
                            config=calibrate_general.template(q,search_family,amin,amax,controls=4+2*min(attempt,2))
                            with progress.phase('general_smile_calibration',attempt=attempt+1,endpoint=endpoint):
                                proposer=calibrate_general.Proposal(config,q,search_family.center(),threads=threads,max_evaluations=fit_evaluations,max_seconds=fit_seconds,progress=progress,backend=backend)
                                config,diagnostics=proposer.fit(endpoint)
                            save_json(candidate/'fit.json',diagnostics);save_json(candidate/'candidate.json',config)
                            proof=fixed_model.run(config,out/'normalized.csv',candidate,progress_mode=progress_mode,heartbeat=heartbeat)
                    except (ValueError,OverflowError,RuntimeError) as error:
                        save_json(candidate/'unresolved.json',dict(reason=str(error),certified=False))
                        progress.event('general_model_candidate','unresolved',attempt=attempt+1,reason=str(error))
                        continue
                    if proof['all_quotes_compatible']:
                        target=out/'bank'/f'model_{len(entries):03d}';tracked_model(candidate,target)
                        theorem=general_theorems.regions(config,q,proof)
                        entries.append(dict(directory=target.relative_to(out).as_posix(),theorems=theorem))
                        if general_theorems.covers(family,theorem):break
                    else:progress.event('general_model_candidate','unresolved',attempt=attempt+1)
            save_json(index_path,entries)
        bank,entries=load_bank(out,q,amin,amax,progress,use_cache=True)
    else:
        (out/'bank').mkdir(exist_ok=True);save_json(out/'bank/index.json',[])
    used=0
    def data_split(fam):
        if not bank:return None
        latest=max(z.t for z in q)
        if fam.coordinates=='left':
            if fam.intervals[0][0]<latest<fam.intervals[0][1]:return fam.split(0,latest),'last_quote_maturity'
        else:
            cut=latest+fam.intervals[2][1]
            if fam.intervals[0][0]<cut<fam.intervals[0][1]:return fam.split(0,cut),'last_quote_maturity'
        for e in entries:
            width=e['theorems']['small_time_width'] if e['theorems'] else None
            if width and width['binary_exponent']==0:
                cut=F(width['coefficient'])/2
                if fam.intervals[2][0]<cut<fam.intervals[2][1]:return fam.split(2,cut),'certified_quote_slack_threshold'
        return None
    def visit(fam,path,depth,budget):
        nonlocal used
        folder=out/'nodes'/path;node_path=folder/'node.json';folder.mkdir(parents=True,exist_ok=True)
        if resume and node_path.exists():
            saved=json.loads(node_path.read_text(encoding='utf-8'))
            check_node(out,saved,fam,q,amin,amax,tolerance,bank,entries,progress)
            progress.event('frontier_node','reused',path=path);return saved
        progress.event('frontier_node','start',path=path,depth=depth,family=fam.json())
        proof=analytic_proof(fam,bank,entries,amin,amax)
        if proof:
            node=dict(family=fam.json(),path=path,status='exact',analytic=proof,lower=str(amin),upper=str(amax))
        elif budget>1 and depth<max_depth and data_split(fam) is not None:
            split,reason=data_split(fam);axis,middle,left,right=split
            node=dict(family=fam.json(),path=path,status='split',axis=axis,middle=str(middle),partition_reason=reason,
                children=[visit(left,path+'0',depth+1,budget//2),visit(right,path+'1',depth+1,budget-budget//2)])
        else:
            records=model_bounds(bank,fam)
            outer_directory='nodes/'+path+'/outer';numeric_reason=None
            retry=False
            if retry_unresolved and (folder/'outer/adaptive.json').exists():
                retry=any(r['kind']=='numeric_search_unresolved' for r in json.loads((folder/'outer/adaptive.json').read_text(encoding='utf-8'))['records'])
            if (folder/'outer/adaptive.json').exists() and not retry:outer=adaptive_outer.verify(q,fam,amin,amax,folder/'outer',progress)
            else:
                try:
                    outer=adaptive_outer.run(q,fam,amin,amax,folder/'outer',strike_cells,target_cells,steps=outer_steps,time_limit=lp_seconds,
                        refinements=mesh_refinements,progress=progress)
                except (OverflowError,ValueError,RuntimeError) as exc:
                    outer=dict(lower=str(amin),upper=str(amax),empty_class_certified=False)
                    outer_directory=None;numeric_reason=str(exc)
                    progress.event('general_outer_search','unresolved',reason=numeric_reason,path=path)
            if outer['empty_class_certified'] and bank:raise ArithmeticError('Outer empty-class proof contradicts a verified model')
            bound=gaps(F(outer['lower']),F(outer['upper']),records,tolerance)
            node=dict(family=fam.json(),path=path,status='certified' if bound['closed'] else 'open',
                lower=outer['lower'],upper=outer['upper'],models=records,gaps=bound,outer_directory=outer_directory)
            if numeric_reason is not None:node['numeric_search_reason']=numeric_reason
            if outer['empty_class_certified']:node['status']='empty_class'
            if not bound['closed'] and not outer['empty_class_certified'] and depth<max_depth and budget>1:
                split=fam.split()
                if split:
                    axis,middle,left,right=split
                    node.update(status='split',axis=axis,middle=str(middle),children=[visit(left,path+'0',depth+1,budget//2),visit(right,path+'1',depth+1,budget-budget//2)])
        save_json(node_path,node);used+=1
        progress.event('frontier_node','done',done=used,path=path,status_detail=node['status'])
        save_json(out/'partial.json',dict(settings=settings,completed_nodes=used,last_node=path))
        return node
    if contradiction is not None:tree=dict(family=family.json(),path='r',status='empty_class',quick_contradiction=contradiction)
    else:tree=visit(family,'r',0,max_leaves)
    # An empty physical class does not have a pair of identification endpoints.
    def empty_leaves(node):
        if node['status']=='split':return [x for c in node['children'] for x in empty_leaves(c)]
        return [node] if node['status']=='empty_class' else []
    empties=empty_leaves(tree)
    if empties and contradiction is None:contradiction=dict(kind='exact_outer_phase_I',path=empties[0]['path'])
    result=dict(version=__version__,settings=settings,tree=tree,contradiction=contradiction,**final_status(tree,bank,contradiction))
    save_json(out/'frontier.json',result)
    page='<!doctype html><meta charset="utf-8"><title>Frontiere generale</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto}pre{white-space:pre-wrap}</style><h1>Frontiere conditionnelle aux cotations fournies</h1><p>'+html.escape(result['status'])+'</p><p>Une boite certifie tous ses parametres. Les feuilles ouvertes restent ouvertes. Aucun calcul fini ne ferme tous les parametres positifs et tous les smiles.</p><pre>'+html.escape(json.dumps(result,indent=2))+'</pre>'
    (out/'frontier.html').write_text(page,encoding='utf-8')
    owned=['normalized.csv','bank/index.json','frontier.json','frontier.html']
    for e in entries:
        folder=out/e['directory'];mname='model_manifest.json' if (folder/'model_manifest.json').exists() else 'exact_manifest.json'
        doc=json.loads((folder/mname).read_text(encoding='utf-8'));owned.append(e['directory']+'/'+mname)
        owned.extend(e['directory']+'/'+n for n in doc['artifacts'])
    for path in sorted((out/'nodes').rglob('*')) if (out/'nodes').exists() else []:
        if path.is_file() and path.suffix in ('.json','.npz'):owned.append(path.relative_to(out).as_posix())
    save_json(manifest,dict(status='completed',settings=settings,artifacts={n:sha256(out/n) for n in owned}))
    return result


def check_node(out,node,family,quotes,amin,amax,tolerance,bank,entries,progress=QUIET):
    if node['family']!=family.json():raise ValueError('Different kernel family in a cover node')
    path=node['path']
    if not path.startswith('r') or any(c not in '01' for c in path[1:]):raise ValueError('Invalid node path')
    status=node['status']
    if 'quick_contradiction' in node:
        proof=general_theorems.quick_infeasibility(quotes,amin,amax)
        if proof is None or proof!=node['quick_contradiction'] or status!='empty_class':raise ValueError('Incorrect quote contradiction')
        return
    proof=analytic_proof(family,bank,entries,amin,amax)
    if status=='exact':
        expected=dict(family=family.json(),path=path,status='exact',analytic=proof,lower=str(amin),upper=str(amax))
        if proof is None or node!=expected:raise ValueError('Unsupported analytic family claim')
        return
    if status=='split' and 'partition_reason' in node:
        axis=int(node['axis']);middle=F(node['middle']);split=family.split(axis,middle)
        expected=dict(family=family.json(),path=path,status='split',axis=axis,middle=str(middle),
            partition_reason=node['partition_reason'],children=node['children'])
        if len(node['children'])!=2 or node!=expected:raise ValueError('Invalid theorem-guided partition')
        for suffix,child,fam in zip(('0','1'),node['children'],split[2:]):
            if child['path']!=path+suffix:raise ValueError('Incorrect child path')
            check_node(out,child,fam,quotes,amin,amax,tolerance,bank,entries,progress)
        return
    directory='nodes/'+path+'/outer'
    if node['outer_directory'] is None:
        if not isinstance(node.get('numeric_search_reason'),str):raise ValueError('Missing failed search explanation')
        outer=dict(lower=str(amin),upper=str(amax),empty_class_certified=False)
    else:
        if node['outer_directory']!=directory:raise ValueError('Invalid outer certificate directory')
        outer=adaptive_outer.verify(quotes,family,amin,amax,out/directory,progress)
    records=model_bounds(bank,family);bound=gaps(F(outer['lower']),F(outer['upper']),records,tolerance)
    expected=dict(family=family.json(),path=path,status='empty_class' if outer['empty_class_certified'] else 'certified' if bound['closed'] else 'open',
        lower=outer['lower'],upper=outer['upper'],models=records,gaps=bound,outer_directory=node['outer_directory'])
    if node['outer_directory'] is None:expected['numeric_search_reason']=node['numeric_search_reason']
    if outer['empty_class_certified'] and bank:raise ValueError('Infeasibility contradicts a verified diffusion')
    if status=='split':
        if bound['closed'] or outer['empty_class_certified']:raise ValueError('Incorrect split status')
        axis=int(node['axis']);middle=F(node['middle']);split=family.split(axis,middle)
        if split is None or split[1]!=middle or len(node['children'])!=2:raise ValueError('Invalid parameter partition')
        expected.update(status='split',axis=axis,middle=str(middle),children=node['children'])
        for suffix,child,fam in zip(('0','1'),node['children'],split[2:]):
            if child['path']!=path+suffix:raise ValueError('Incorrect child path')
            check_node(out,child,fam,quotes,amin,amax,tolerance,bank,entries,progress)
    if node!=expected:raise ValueError('Incorrect posterior family summary')


def verify(output,progress=QUIET):
    out=Path(output);m=json.loads((out/'general_manifest.json').read_text(encoding='utf-8'))
    if m['status']!='completed':raise ValueError('Resume the incomplete calculation first')
    for name,digest in m['artifacts'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or sha256(out/p)!=digest:raise ValueError('Changed general frontier artifact: '+name)
    settings=m['settings'];q=read_quotes(out/'normalized.csv');family=Family.make(settings['family'])
    amin,amax,tol=map(F,(settings['amin'],settings['amax'],settings['tolerance']))
    if quote_payload(q)!=settings['quotes']:raise ValueError('Different input smile')
    bank,entries=load_bank(out,q,amin,amax,progress)
    result=json.loads((out/'frontier.json').read_text(encoding='utf-8'));tree=result['tree']
    if tree['path']!='r':raise ValueError('Missing parameter-domain root')
    check_node(out,tree,family,q,amin,amax,tol,bank,entries,progress)
    quick=general_theorems.quick_infeasibility(q,amin,amax)
    def empty(node):
        if node['status']=='split':return [x for c in node['children'] for x in empty(c)]
        return [node] if node['status']=='empty_class' else []
    leaves=empty(tree);contradiction=quick
    if leaves and quick is None:contradiction=dict(kind='exact_outer_phase_I',path=leaves[0]['path'])
    expected=dict(version=settings['version'],settings=settings,tree=tree,contradiction=contradiction,**final_status(tree,bank,contradiction))
    if result!=expected:raise ValueError('Incorrect domain or universal completion claim')
    return dict(verified=True,status=result['status'],model_class_empty_certified=result['model_class_empty_certified'],
        all_parameters_in_requested_domain_solved=result['all_parameters_in_requested_domain_solved'],
        all_positive_parameters_solved=False,all_smiles_solved=False,optimizer_restarted=False,pde_solver_restarted=False)
