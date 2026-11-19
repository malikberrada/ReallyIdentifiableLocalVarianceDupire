"""Target finite, explicit domains with new certified endpoint diffusions."""
from pathlib import Path
from fractions import Fraction as F
import json
import html
import re
from . import clock_models, fixed_model, general_frontier, calibrate_general
from .families import Family
from .io import read_quotes, write_quotes, quote_payload, save_json, sha256
from .progress import Progress, QUIET


def specifications(path):
    doc=json.loads(Path(path).read_text(encoding='utf-8'))
    items=doc['targets'];names=[];answer=[]
    for item in items:
        name=item['name']
        if not re.fullmatch('[A-Za-z0-9_-]{1,60}',name) or name in names:
            raise ValueError('Target names must be distinct safe directory names')
        names.append(name);answer.append(dict(name=name,family=Family.make(item['family']).json()))
    if not answer:raise ValueError('Supply at least one target family')
    return answer


def run(quotes_file,targets_file,output,model_runs=(),amin='1/400',amax='1/4',
        backend='auto',threads=1,fit_evaluations=1500,fit_seconds=180,controls=6,
        spatial_cells=128,pulse_cells=16,time_steps=512,refinements=2,
        progress_mode='plain',heartbeat=2,resume=False,**outer_options):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    quotes=read_quotes(quotes_file);targets=specifications(targets_file)
    amin,amax=F(amin),F(amax)
    if not 0<amin<amax:raise ValueError('Positive ordered variance priors required')
    settings=dict(quotes=quote_payload(quotes),targets=targets,amin=str(amin),amax=str(amax),
        backend=backend,threads=threads,fit_evaluations=fit_evaluations,fit_seconds=fit_seconds,
        controls=controls,spatial_cells=spatial_cells,pulse_cells=pulse_cells,time_steps=time_steps,
        refinements=refinements,outer_options=outer_options)
    manifest=out/'target_manifest.json';progress=Progress(progress_mode,heartbeat,out/'events.jsonl')
    if manifest.exists():
        previous=json.loads(manifest.read_text(encoding='utf-8'))
        if not resume or previous['settings']!=settings:raise ValueError('Use --resume with identical settings or another output')
        if previous['status']=='completed':
            verify(out,progress);return json.loads((out/'targets.json').read_text(encoding='utf-8'))
    save_json(manifest,dict(status='running',settings=settings));write_quotes(out/'normalized.csv',quotes)
    save_json(out/'domains.json',dict(targets=targets))
    imported=out/'imported';imported.mkdir(exist_ok=True)
    if not (imported/'index.json').exists():
        sources=[]
        for i,source in enumerate(model_runs):
            target=imported/f'model_{i:03d}'
            general_frontier.tracked_model(source,target);fixed_model.verify_cached(target,progress)
            sources.append(target.relative_to(out).as_posix())
        save_json(imported/'index.json',sources)
    sources=[out/n for n in json.loads((imported/'index.json').read_text(encoding='utf-8'))]
    initial_bank=[];entries=[]
    for source in sources:
        fixed_model.verify_cached(source,progress);model=fixed_model.Model(source)
        if quote_payload(model.quotes)!=quote_payload(quotes) or F(model.config['amin'])!=amin or F(model.config['amax'])!=amax:
            raise ValueError('Imported target model uses another quote or variance class')
        initial_bank.append(model)
        from .general_theorems import regions
        entries.append(dict(theorems=regions(model.config,quotes,model.pricing)))
    with progress.phase('clock_variance_budgets'):
        ranges=clock_models.safe_variance_ranges(quotes,amin,amax)
    records=[]
    for i,spec in enumerate(targets):
        name=spec['name'];family=Family.make(spec['family']);folder=out/name;folder.mkdir(exist_ok=True)
        progress.event('target_domain','start',done=i,total=len(targets),name=name,family=family.json())
        candidate_sources=[];attempts=[]
        covered=general_frontier.analytic_proof(family,initial_bank,entries,amin,amax)
        if not covered:
            for endpoint in ('minimum','maximum'):
                candidate=folder/'candidates'/endpoint
                if (candidate/'model_manifest.json').exists():
                    saved=json.loads((candidate/'model_manifest.json').read_text(encoding='utf-8'))
                    if saved['compatible']:
                        fixed_model.verify_cached(candidate,progress);candidate_sources.append(candidate)
                    attempts.append(dict(endpoint=endpoint,kind='resumed',compatible=saved['compatible']));continue
                cfg=clock_models.construct(quotes,family,amin,amax,endpoint,progress,ranges=ranges) if ranges is not None else None
                if cfg is not None:
                    clock_models.run(cfg,out/'normalized.csv',candidate);candidate_sources.append(candidate)
                    attempts.append(dict(endpoint=endpoint,kind='exact_analytic_clock',compatible=True));continue
                seed=initial_bank[0].config if initial_bank else None
                cfg=calibrate_general.target_template(quotes,family,amin,amax,endpoint,controls,seed)
                try:
                    with progress.phase('target_endpoint_fit',name=name,endpoint=endpoint):
                        proposer=calibrate_general.Proposal(cfg,quotes,family.center(),threads=threads,
                            max_evaluations=fit_evaluations,max_seconds=fit_seconds,progress=progress,backend=backend)
                        # The plateau is locked exactly. Only compensation is fitted.
                        cfg,diagnostics=proposer.fit()
                    save_json(candidate/'fit.json',diagnostics);save_json(candidate/'candidate.json',cfg)
                    proof=fixed_model.run(cfg,out/'normalized.csv',candidate,spatial_cells,pulse_cells,
                        time_steps,refinements,progress_mode,heartbeat)
                    compatible=proof['all_quotes_compatible']
                    attempts.append(dict(endpoint=endpoint,kind='spatial_plateau_PDE',compatible=compatible))
                    if compatible:candidate_sources.append(candidate)
                except (RuntimeError,ValueError,OverflowError) as error:
                    save_json(candidate/'unresolved.json',dict(reason=str(error),certified=False))
                    attempts.append(dict(endpoint=endpoint,kind='unresolved_proposal',compatible=False,reason=str(error)))
                    progress.event('target_endpoint','unresolved',name=name,endpoint=endpoint,reason=str(error))
        # Keep imported models when either new endpoint remains unresolved.
        model_sources=candidate_sources if len(candidate_sources)==2 else sources+candidate_sources
        result=general_frontier.run(out/'normalized.csv',folder/'frontier',family,amin=str(amin),amax=str(amax),
            model_runs=model_sources,fit_models=0,threads=threads,backend=backend,
            progress_mode=progress_mode,heartbeat=heartbeat,
            resume=resume and (folder/'frontier/general_manifest.json').exists(),**outer_options)
        records.append(dict(name=name,directory=name+'/frontier',attempts=attempts,
            status=result['status'],certified_leaf_boxes=result['certified_leaf_boxes'],
            open_leaf_boxes=result['open_leaf_boxes'],requested_domain_solved=result['all_parameters_in_requested_domain_solved']))
        save_json(out/'partial.json',dict(settings=settings,completed=records))
        progress.event('target_domain','done',done=i+1,total=len(targets),name=name,status_detail=result['status'])
    result=dict(settings=settings,targets=records,all_requested_targets_solved=all(r['requested_domain_solved'] for r in records),
        all_positive_parameters_solved=False,all_smiles_solved=False,
        arithmetic='rational/CPU certificates; binary64 CPU or CUDA numerical proposals')
    save_json(out/'targets.json',result)
    rows=''.join('<tr><td>'+html.escape(r['name'])+'</td><td>'+html.escape(r['status'])+'</td><td>'+str(r['open_leaf_boxes'])+'</td></tr>' for r in records)
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Domaines cibles</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}</style><h1>Domaines cibles certifies</h1><p>Une preuve porte sur une famille continue de fenetres et sur les cotations fournies. Les domaines ouverts restent ouverts.</p><table><tr><th>Domaine</th><th>Statut</th><th>Boites ouvertes</th></tr>'+rows+'</table>',encoding='utf-8')
    owned=['normalized.csv','domains.json','targets.json','report.html']
    for r in records:
        sub=json.loads((out/r['directory']/'general_manifest.json').read_text(encoding='utf-8'))
        owned.append(r['directory']+'/general_manifest.json')
        owned.extend(r['directory']+'/'+n for n in sub['artifacts'])
    save_json(manifest,dict(status='completed',settings=settings,artifacts={n:sha256(out/n) for n in owned}))
    return result


def verify(output,progress=QUIET):
    out=Path(output);m=json.loads((out/'target_manifest.json').read_text(encoding='utf-8'))
    if m['status']!='completed':raise ValueError('Incomplete targeted domains')
    for name,digest in m['artifacts'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or sha256(out/p)!=digest:raise ValueError('Changed target artifact: '+name)
    result=json.loads((out/'targets.json').read_text(encoding='utf-8'));settings=m['settings']
    if result['settings']!=settings or quote_payload(read_quotes(out/'normalized.csv'))!=settings['quotes']:
        raise ValueError('Different target observations or settings')
    if len(result['targets'])!=len(settings['targets']):raise ValueError('Missing target')
    solved=[]
    for spec,row in zip(settings['targets'],result['targets']):
        if row['name']!=spec['name'] or row['directory']!=spec['name']+'/frontier':raise ValueError('Different target domain')
        sub=out/row['directory'];checked=general_frontier.verify(sub,progress)
        frontier=json.loads((sub/'frontier.json').read_text(encoding='utf-8'))
        if frontier['settings']['family']!=spec['family'] or frontier['settings']['quotes']!=settings['quotes']:
            raise ValueError('Target certificate uses another family or observations')
        if any(row[k]!=frontier[k] for k in ('status','certified_leaf_boxes','open_leaf_boxes')):
            raise ValueError('Incorrect target frontier summary')
        if F(frontier['settings']['amin'])!=F(settings['amin']) or F(frontier['settings']['amax'])!=F(settings['amax']):
            raise ValueError('Different target variance class')
        if row['requested_domain_solved']!=checked['all_parameters_in_requested_domain_solved']:
            raise ValueError('Incorrect target closure claim')
        solved.append(row['requested_domain_solved'])
    if result['all_requested_targets_solved']!=all(solved) or result['all_positive_parameters_solved'] or result['all_smiles_solved']:
        raise ValueError('False target or universal completion claim')
    return dict(verified=True,targets=len(solved),all_requested_targets_solved=all(solved),
                all_positive_parameters_solved=False,all_smiles_solved=False,
                optimizer_restarted=False,pde_solver_restarted=False)
