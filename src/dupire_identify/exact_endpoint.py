"""Exact endpoint certificates from fixed-coefficient plateau diffusions."""
from fractions import Fraction as F
from pathlib import Path
import html
import json
import numpy as np
from . import __version__, plateau, denominator, time_models
from .continuous import Window
from .interval import I
from .io import read_quotes,write_quotes,quote_payload,save_json,sha256
from .progress import Progress,QUIET


def contains(rect,w):
    return (F(rect['t_left'])<=w.t-w.ht and w.t+w.ht<=F(rect['t_right']) and
            F(rect['k_left'])<=w.k-w.hk and w.k+w.hk<=F(rect['k_right']))


def global_claim(config,quotes,w,pricing,floor,minimum,version=None):
    lo,hi=plateau.validate_config(config)
    rectangles=plateau.plateau_rectangles(config,hi)
    maximum=pricing['all_quotes_compatible'] and floor>0 and any(contains(r,w) for r in rectangles)
    min_exact=minimum is not None and minimum['average']==I(lo).json()
    return dict(version=version or __version__,status=('exact_global_endpoints_certified' if maximum and min_exact else
        'exact_global_maximum_certified' if maximum else 'maximum_not_certified'),
        quotes=quote_payload(quotes),window=w.json(),amin=str(lo),amax=str(hi),
        maximum_exact_proved=bool(maximum),minimum_exact_proved=bool(min_exact),
        minimum=I(lo).json() if min_exact else None,maximum=I(hi).json() if maximum else None,
        outer_prior=I(lo,hi).json(),maximum_gap='0' if maximum else None,
        denominator_lower=str(floor),maximum_rectangles=rectangles,
        model_kind='fixed continuous-path diffusion; C2 in state, piecewise constant in calendar time',
        maximum_theorem='A<=amax for every admissible model; this compatible diffusion has a=amax on the full kernel support',
        all_positive_half_line_controlled=True,unbounded_parameter_frontier_solved=False)


def regions(config,quotes,minimum):
    lo,hi=plateau.validate_config(config);patches=[]
    if minimum is not None:
        for rect in plateau.plateau_rectangles(config,hi):
            for left,right,c in minimum['segments']:
                if F(c[0])!=lo or any(F(v) for v in c[1:]):continue
                a=max(F(left),F(rect['t_left']));b=min(F(right),F(rect['t_right']))
                if a<b:
                    patches.append(dict(rect,t_left=str(a),t_right=str(b),
                        source='plateau_diffusion_and_exact_time_change',
                        identification_interval=[str(lo),str(hi)],minimax_absolute_error=str((hi-lo)/2)))
    analytic=None
    try:analytic=time_models.full_prior_region(quotes,F(config['prefix_variance']),lo,hi)
    except ValueError:pass
    return dict(rectangles=patches,temporal_strips=analytic,
        scope='Every nonnegative nonzero kernel supported inside a certified rectangle; the specified bump kernels in temporal strips',
        unbounded_parameter_frontier_solved=False)


def manifest(out,names,settings):
    save_json(out/'exact_manifest.json',dict(status='completed',settings=settings,
        artifacts={name:sha256(out/name) for name in names}))


def run(quotes_file,output,w,config_file=None,initial_run=None,spatial_cells=192,pulse_cells=32,
        time_steps=512,refinements=2,progress_mode='plain',heartbeat=2,resume=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    progress=Progress(progress_mode,heartbeat,out/'events.jsonl')
    quotes=read_quotes(quotes_file)
    config=json.loads(Path(config_file).read_text(encoding='utf-8')) if config_file else plateau.default_config(quotes,w)
    lo,hi=plateau.validate_config(config)
    settings=dict(version=__version__,quotes=quote_payload(quotes),window=w.json(),config=config,
        spatial_cells=spatial_cells,pulse_cells=pulse_cells,time_steps=time_steps,refinements=refinements)
    if refinements<1 or spatial_cells<16 or pulse_cells<4 or time_steps<16:raise ValueError('Positive certified refinement budgets required')
    if (out/'exact_manifest.json').exists():
        if not resume:raise ValueError('Exact endpoint run exists; use --resume or another output')
        old=json.loads((out/'exact_manifest.json').read_text(encoding='utf-8'))
        if old['settings']!=settings:raise ValueError('Resume requires identical settings')
        verify(out,progress)
        return json.loads((out/'result.json').read_text(encoding='utf-8'))
    checkpoint=None
    if (out/'checkpoint.json').exists():
        if not resume:raise ValueError('Interrupted run exists; use --resume')
        checkpoint=json.loads((out/'checkpoint.json').read_text(encoding='utf-8'))
        if checkpoint['settings']!=settings:raise ValueError('Checkpoint uses different settings')
        for name in ('surface.npz','mesh.json','diffusion.json'):
            if sha256(out/name)!=checkpoint['hashes'][name]:raise ValueError('Changed checkpoint: '+name)
    write_quotes(out/'normalized.csv',quotes);save_json(out/'diffusion.json',config)
    if initial_run and (Path(initial_run)/'denominator_floor.json').exists():
        floor_doc=json.loads((Path(initial_run)/'denominator_floor.json').read_text(encoding='utf-8'))
        floor=denominator.verify_bundle(floor_doc,quotes,w,lo,hi,progress)
    else:
        brown=denominator.brownian_floor(w,lo,hi);floor=F(brown['dyadic_lower'])
        floor_doc=dict(kind='global_denominator_bundle_v1',window=w.json(),amin=str(lo),amax=str(hi),
            brownian=brown,static_subsolutions=[],lower_bound=str(floor),approximate=float(floor),positive=True)
    save_json(out/'denominator_floor.json',floor_doc)
    minimum=None
    candidates=time_models.candidates(quotes,w,F(config['prefix_variance']),lo,hi,adaptive_ramps=True)
    for candidate in candidates:
        if candidate['average']==I(lo).json():
            time_models.verify(candidate,quotes,w);minimum=candidate;break
    save_json(out/'minimum_model.json',minimum)
    pricing=None
    for level in range(checkpoint['level'] if checkpoint else 0,refinements):
        sc,pc,steps=spatial_cells*2**level,pulse_cells*2**level,time_steps*2**level
        progress.event('exact_endpoint_refinement','start',done=level,total=refinements,
            spatial_cells=sc,pulse_cells=pc,time_steps=steps)
        if checkpoint is not None and level==checkpoint['level']:
            mesh=json.loads((out/'mesh.json').read_text(encoding='utf-8'));grid=list(map(F,mesh['grid']));times=mesh['times']
            with np.load(out/'surface.npz',allow_pickle=False) as loaded:arrays={k:loaded[k].copy() for k in loaded.files}
            progress.event('plateau_checkpoint','reused',level=level)
            checkpoint=None
        else:
            grid=plateau.make_grid(config,sc,pc)
            with progress.phase('plateau_proposal',engine=plateau.proposal_engine()+' compact SDIRK2; CPU',nodes=len(grid)):
                arrays,times=plateau.solve_numerical(config,grid,steps,progress)
            np.savez_compressed(out/'surface.tmp.npz',**arrays);(out/'surface.tmp.npz').replace(out/'surface.npz')
            save_json(out/'mesh.json',dict(grid=list(map(str,grid)),times=times))
            save_json(out/'checkpoint.json',dict(settings=settings,level=level,
                hashes={name:sha256(out/name) for name in ('surface.npz','mesh.json','diffusion.json')}))
        with progress.phase('continuous_price_proof',arithmetic='outward Bernstein + rational endpoints'):
            pricing=plateau.certify_prices(config,grid,times,arrays,quotes,progress)
        save_json(out/'pricing.json',pricing)
        progress.event('exact_endpoint_refinement','done',done=level+1,total=refinements,
            all_quotes_compatible=pricing['all_quotes_compatible'])
        if pricing['all_quotes_compatible']:break
    result=global_claim(config,quotes,w,pricing,floor,minimum)
    save_json(out/'result.json',result);atlas=regions(config,quotes,minimum);save_json(out/'regions.json',atlas)
    maximum=dict(model_type='certified_plateau_diffusion',diffusion='diffusion.json',pricing='pricing.json',
        average=result['maximum'],certified=result['maximum_exact_proved'],window=w.json(),quotes=quote_payload(quotes))
    save_json(out/'maximum_model.json',maximum)
    rows=''.join('<tr><td>'+html.escape(r['id'])+'</td><td>'+html.escape(str(r['price']['approximate']))+'</td><td>'+str(r['within_bid_ask'])+'</td></tr>' for r in pricing['prices'])
    text='<!doctype html><html lang="fr"><meta charset="utf-8"><title>Maximum exact</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto}td,th{border:1px solid #ccd;padding:10px}table{border-collapse:collapse}pre{white-space:pre-wrap}</style><h1>Maximum global par plateau exact</h1><p>'+html.escape(result['status'])+'</p><p>Le coefficient de la diffusion, et non la surface numerique, definit le modele. Les erreurs de prix ci-dessous sont controlees par le residu de Dupire sur tout le domaine et par comparaison Black aux frontieres.</p><pre>'+html.escape(json.dumps(result,indent=2))+'</pre><h2>Prix continus encadres</h2><table><tr><th>Contrat</th><th>Intervalle de prix</th><th>Dans le spread</th></tr>'+rows+'</table><p>La frontiere sur tous les parametres positifs sans domaine borne reste ouverte. Voir regions.json et la commande universal-frontier.</p></html>'
    (out/'report.html').write_text(text,encoding='utf-8')
    owned=['normalized.csv','diffusion.json','surface.npz','mesh.json','pricing.json','denominator_floor.json',
        'minimum_model.json','maximum_model.json','result.json','regions.json','report.html']
    manifest(out,owned,settings)
    return result


def verify(directory,progress=QUIET):
    out=Path(directory);doc=json.loads((out/'exact_manifest.json').read_text(encoding='utf-8'))
    expected={'normalized.csv','diffusion.json','surface.npz','mesh.json','pricing.json','denominator_floor.json',
        'minimum_model.json','maximum_model.json','result.json','regions.json','report.html'}
    if doc.get('status')!='completed' or set(doc['artifacts'])!=expected:raise ValueError('Incomplete exact endpoint manifest')
    for name,digest in doc['artifacts'].items():
        if sha256(out/name)!=digest:raise ValueError('Changed exact endpoint artifact: '+name)
    config=json.loads((out/'diffusion.json').read_text(encoding='utf-8'));quotes=read_quotes(out/'normalized.csv')
    settings=doc['settings'];w=Window.make(**settings['window']);lo,hi=plateau.validate_config(config)
    if config!=settings['config'] or quote_payload(quotes)!=settings['quotes']:raise ValueError('Different certified problem')
    floor_doc=json.loads((out/'denominator_floor.json').read_text(encoding='utf-8'))
    floor=denominator.verify_bundle(floor_doc,quotes,w,lo,hi,progress)
    mesh=json.loads((out/'mesh.json').read_text(encoding='utf-8'));grid=list(map(F,mesh['grid']))
    with np.load(out/'surface.npz',allow_pickle=False) as arrays:
        proof=plateau.certify_prices(config,grid,mesh['times'],arrays,quotes,progress)
    if proof!=json.loads((out/'pricing.json').read_text(encoding='utf-8')):raise ValueError('Incorrect continuous price enclosure')
    minimum=json.loads((out/'minimum_model.json').read_text(encoding='utf-8'))
    if minimum is not None:
        if F(minimum['amin'])!=lo or F(minimum['amax'])!=hi:raise ValueError('Minimum uses different variance bounds')
        time_models.verify(minimum,quotes,w)
    if settings['version'] not in ('0.4.0','0.5.0'):raise ValueError('Unsupported exact certificate version')
    result=global_claim(config,quotes,w,proof,floor,minimum,settings['version'])
    if result!=json.loads((out/'result.json').read_text(encoding='utf-8')):raise ValueError('Incorrect exact endpoint claim')
    maximum=dict(model_type='certified_plateau_diffusion',diffusion='diffusion.json',pricing='pricing.json',
        average=result['maximum'],certified=result['maximum_exact_proved'],window=w.json(),quotes=quote_payload(quotes))
    if maximum!=json.loads((out/'maximum_model.json').read_text(encoding='utf-8')):raise ValueError('Incorrect maximum export')
    atlas=regions(config,quotes,minimum)
    if atlas!=json.loads((out/'regions.json').read_text(encoding='utf-8')):raise ValueError('Incorrect continuum regions')
    return dict(verified=True,maximum_exact_proved=result['maximum_exact_proved'],minimum_exact_proved=result['minimum_exact_proved'],
        maximum=result['maximum'],minimum=result['minimum'],continuous_prices_recomputed=True,
        global_variance_bounds_verified=True,optimizer_restarted=False,pde_solver_restarted=False,
        unbounded_parameter_frontier_solved=False)
