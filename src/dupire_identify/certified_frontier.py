"""Finite requested frontier with per-window global gap certificates.

Completion on a requested grid and a theorem on a temporal-strip continuum
are separate claims. Unresolved windows and exhausted budgets stay explicit.
"""
from itertools import product
from pathlib import Path
from fractions import Fraction as F
import json
import html
from .continuous import Window
from .io import read_quotes,write_quotes,quote_payload,save_json,sha256,rational
from . import __version__,tightening,time_models,inner
from .progress import Progress,QUIET


def run(quotes_file,output,t,k,ht,hk,amin='1/400',amax='1/4',outer_levels=(8,),
        spline_levels=(8,16),denominator_levels=(8,),subdivisions=2,quadrature_cells=32,
        tolerance='1/100',baseline_variances=('1/25',),time_limit=120,progress_mode='plain',
        heartbeat=2,resume=False,proposal_solver='full',expand_marginals=False,initial_run=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    q=read_quotes(quotes_file);amin,amax,tolerance=map(rational,(amin,amax,tolerance))
    windows=[Window.make(*x) for x in product(t,k,ht,hk)]
    if not windows:raise ValueError('At least one requested window required')
    if len({tuple(w.json().values()) for w in windows})!=len(windows):raise ValueError('Duplicate requested windows')
    progress=Progress(progress_mode,heartbeat,out/'frontier_events.jsonl')
    settings=dict(version=__version__,quotes=quote_payload(q),windows=[w.json() for w in windows],
        amin=str(amin),amax=str(amax),outer_levels=list(outer_levels),spline_levels=list(spline_levels),
        denominator_levels=list(denominator_levels),subdivisions=subdivisions,quadrature_cells=quadrature_cells,
        tolerance=str(tolerance),baseline_variances=[str(rational(x)) for x in baseline_variances],lp_time_limit=time_limit,
        proposal_solver=proposal_solver,expand_marginals=bool(expand_marginals),initial_run=str(initial_run) if initial_run else None)
    manifest_path=out/'frontier_manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text(encoding='utf-8'))
        if not resume:raise ValueError('Frontier exists; use --resume')
        if old['settings']!=settings:raise ValueError('Frontier resume requires identical settings')
    manifest=dict(settings=settings,status='running',artifacts={})
    save_json(manifest_path,manifest);write_quotes(out/'normalized.csv',q)
    region=None
    if len(baseline_variances)==1:
        try:region=time_models.full_prior_region(q,rational(baseline_variances[0]),amin,amax)
        except ValueError:pass
    rows=[];sources=[];seed_files=[];core_cache={}
    if initial_run:
        path=Path(initial_run)
        source=[json.loads((path/name).read_text(encoding='utf-8')) for name in ('minimum_model.json','maximum_model.json')] if path.is_dir() else json.loads(path.read_text(encoding='utf-8'))
        if isinstance(source,dict):source=[source]
        sources=[doc for doc in source if 'theta' in doc]
    def exact_strip(w):
        if region is None:return False
        if w.t-w.ht>F(region['after_last_maturity']['strict_time_support_left_lower']):return True
        return any(F(strip['left'])<w.t-w.ht and w.t+w.ht<F(strip['right']) and
                   2*w.ht<F(strip['strict_time_support_width_upper']) for strip in region['strips'])
    try:
        for index,w in enumerate(windows):
            folder=f'window_{index:04d}';target=out/folder
            progress.event('frontier_window','start',done=index,total=len(windows),window=w.json())
            seed_path=None
            if sources and not exact_strip(w):
                docs=[inner.retarget(doc,q,w,quadrature_cells,progress,core_cache) for doc in sources]
                seed_path=out/'seeds'/f'window_{index:04d}.json';save_json(seed_path,docs)
                seed_files.append(seed_path.relative_to(out).as_posix())
            result=tightening.run(out/'normalized.csv',target,w,amin,amax,outer_levels,spline_levels,
                subdivisions,quadrature_cells,tolerance,baseline_variances,time_limit=time_limit,
                progress_mode=progress_mode,heartbeat=heartbeat,
                resume=resume and (target/'strong_manifest.json').exists(),
                positive_denominator=True,denominator_levels=denominator_levels,expanded_models=True,
                proposal_solver=proposal_solver,expand_marginals=expand_marginals,
                initial_run=seed_path,verification_cache=core_cache)
            rows.append(dict(window=w.json(),directory=folder,outer=result['outer']['approximate'],
                gap=result['gap'],denominator_lower=result['denominator_floor']['lower_bound'],
                minimum=result['minimum_model']['average'],maximum=result['maximum_model']['average']))
            progress.event('frontier_window','done',done=index+1,total=len(windows),status_detail=result['gap']['status'])
            save_json(out/'frontier_partial.json',dict(settings=settings,completed=len(rows),windows=rows))
        complete=all(row['gap']['endpoints_within_tolerance_certified'] for row in rows)
        result=dict(version=__version__,settings=settings,windows=rows,
            requested_grid_certified=complete,universal_frontier_proved=False,
            status='requested_grid_certified' if complete else 'requested_grid_has_open_gaps',
            continuum_full_prior_region=region,
            scope='all models in the announced continuous class, at the explicitly requested windows and tolerance')
        save_json(out/'frontier.json',result)
        table=''.join('<tr><td>'+html.escape(str(row['window']))+'</td><td>'+str(row['outer'])+'</td><td>'+str(row['gap']['approximate_endpoint_gaps'])+'</td><td><a href="'+row['directory']+'/report.html">'+row['gap']['status']+'</a></td></tr>' for row in rows)
        page='<!doctype html><meta charset="utf-8"><title>Frontiere certifiee</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto}table{border-collapse:collapse}td,th{border:1px solid #ccd;padding:12px}</style><h1>Frontiere sur les fenetres demandees</h1><p>'+html.escape(result['status'])+'</p><p>Chaque ligne porte sur toute la classe continue. Cette grille finie ne couvre pas tous les noyaux possibles.</p><table><tr><th>Fenetre</th><th>Bornes continues</th><th>Ecarts bas / haut</th><th>Certificats</th></tr>'+table+'</table>'
        (out/'frontier.html').write_text(page,encoding='utf-8')
        owned=['normalized.csv','frontier.json','frontier.html']+seed_files
        for row in rows:
            sm=json.loads((out/row['directory']/'strong_manifest.json').read_text(encoding='utf-8'))
            owned.append(row['directory']+'/strong_manifest.json')
            owned.extend(row['directory']+'/'+name for name in sm['artifacts'])
        manifest.update(status='completed',artifacts={name:sha256(out/name) for name in owned})
        save_json(manifest_path,manifest)
        return result
    except BaseException as error:
        manifest.update(status='failed',error=str(error));save_json(manifest_path,manifest);raise


def verify(output,progress=QUIET):
    out=Path(output);manifest=json.loads((out/'frontier_manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed':raise ValueError('Complete the requested frontier first')
    for name,digest in manifest['artifacts'].items():
        if sha256(out/name)!=digest:raise ValueError('Changed frontier artifact: '+name)
    result=json.loads((out/'frontier.json').read_text(encoding='utf-8'))
    settings=manifest['settings'];q=read_quotes(out/'normalized.csv')
    if result['settings']!=settings or quote_payload(q)!=settings['quotes']:raise ValueError('Different frontier problem')
    if [row['window'] for row in result['windows']]!=settings['windows']:raise ValueError('Missing or changed frontier windows')
    complete=True;core_cache={}
    for i,row in enumerate(result['windows']):
        if row['directory']!=f'window_{i:04d}':raise ValueError('Invalid window certificate directory')
        checked=tightening.verify(out/row['directory'],progress,core_cache)
        sub=json.loads((out/row['directory']/'result.json').read_text(encoding='utf-8'))
        for key in ('quotes','amin','amax','tolerance','baseline_variances'):
            if sub['settings'][key]!=settings[key]:raise ValueError('Window uses a different frontier class or tolerance')
        if not checked['denominator_positive_proved']:raise ValueError('Missing positive denominator certificate')
        if (row['window']!=sub['settings']['window'] or row['gap']!=checked['endpoint_gaps'] or
            row['denominator_lower']!=checked['denominator_lower'] or row['outer']!=sub['outer']['approximate'] or
            row['minimum']!=sub['minimum_model']['average'] or row['maximum']!=sub['maximum_model']['average']):
            raise ValueError('Incorrect window frontier summary')
        complete &= checked['endpoint_gaps']['endpoints_within_tolerance_certified']
        progress.event('frontier_verification','running',done=i+1,total=len(result['windows']))
    expected_region=None
    if len(settings['baseline_variances'])==1:
        try:expected_region=time_models.full_prior_region(q,rational(settings['baseline_variances'][0]),F(settings['amin']),F(settings['amax']))
        except ValueError:pass
    if (result['continuum_full_prior_region']!=expected_region or result['universal_frontier_proved'] or
        result['requested_grid_certified']!=complete or result['status']!=('requested_grid_certified' if complete else 'requested_grid_has_open_gaps')):
        raise ValueError('Incorrect frontier completion claim')
    return dict(verified=True,windows=len(result['windows']),requested_grid_certified=bool(complete),
                universal_frontier_proved=False,continuum_region_verified=expected_region is not None,optimizer_restarted=False)
