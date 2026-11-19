"""Coupled continuous outer certificates + exactly feasible diffusion witnesses."""
from fractions import Fraction as F
from pathlib import Path
import html
import json
import numpy as np
from . import __version__
from .io import read_quotes,write_quotes,save_json,sha256,quote_payload,rational
from .continuous import Window,outer_bounds
from .workflow import default_window
from . import occupation,inner,time_models,certlp,denominator
from .progress import Progress,QUIET


def endpoint_gap(lower,upper,models,tolerance):
    if not models:return dict(status='no_certified_witness',endpoints_within_tolerance_certified=False),None,None
    minimum=min(models,key=lambda x:F(x['average']['hi']))
    maximum=max(models,key=lambda x:F(x['average']['lo']))
    low_gap=F(minimum['average']['hi'])-lower
    high_gap=upper-F(maximum['average']['lo'])
    if min(low_gap,high_gap)<0:raise ArithmeticError('Continuous witness contradicts outer bound')
    separation=max(F(0),F(maximum['average']['lo'])-F(minimum['average']['hi']))
    return dict(status='endpoints_within_tolerance' if max(low_gap,high_gap)<=tolerance else 'gap_still_open',
        endpoints_within_tolerance_certified=max(low_gap,high_gap)<=tolerance,
        lower_endpoint_gap=str(low_gap),upper_endpoint_gap=str(high_gap),
        approximate_endpoint_gaps=[float(low_gap),float(high_gap)],
        witnessed_identification_width_lower=str(separation),
        identification_width_upper=str(upper-lower),minimax_error_lower=str(separation/2),
        tolerance=str(tolerance),global_frontier_theorem_proved=False),minimum,maximum


def html_report(path,result):
    def e(x):return html.escape(str(x))
    gap=result['gap'];lower=result['outer']['lower'];upper=result['outer']['upper']
    rows=[('Outer continuous bound',[float(F(lower)),float(F(upper))])]
    for key in ('minimum_model','maximum_model'):
        model=result.get(key)
        if model:rows.append((key,model['average']['approximate']))
    table=''.join('<tr><td>'+e(a)+'</td><td>'+e(b)+'</td></tr>' for a,b in rows)
    text=f'''<!doctype html><html lang="fr"><meta charset="utf-8"><title>Dupire continuous certificates</title>
<style>body{{font:16px system-ui;max-width:1050px;margin:36px auto;padding:20px;color:#172e44;background:#f6f8fb}}
table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:14px;border:1px solid #becbd8}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:18px}}h1{{color:#123e64}}</style>
<h1>Bornes continues et modeles certifies</h1>
<p>Variance annualisee normalisee. Noyau pondere par la densite propre a chaque modele.</p>
<p>Minorant positif du denominateur : {e(result.get('denominator_floor',{}).get('lower_bound','non calcule dans ce mode'))}.</p>
<table><tr><th>Certificat</th><th>Intervalle de variance</th></tr>{table}</table>
<h2>Distance garantie aux extremites</h2><pre>{e(json.dumps(gap,indent=2,ensure_ascii=False))}</pre>
<p>Un ecart ouvert est conserve comme tel. Le primal de la relaxation exterieure n'est pas utilise comme diffusion.</p>
<h2>Hypotheses</h2><pre>{e(json.dumps(result['settings'],indent=2,ensure_ascii=False))}</pre>
</html>'''
    Path(path).write_text(text,encoding='utf-8')


def run(quotes_file,output,window=None,amin='1/400',amax='1/4',outer_levels=(8,),
        spline_levels=(12,16),subdivisions=2,quadrature_cells=24,tolerance='1/100',
        baseline_variances=('1/25',),density_floor='1/100',time_limit=120,
        progress_mode='plain',heartbeat=2,resume=False,overwrite=False,
        positive_denominator=False,denominator_levels=(8,),expanded_models=False,initial_run=None,
        expand_marginals=False,proposal_solver='auto',verification_cache=None):
    if verification_cache is None:verification_cache={}
    if initial_run and not Path(initial_run).exists():raise ValueError('Initial model path does not exist: '+str(initial_run))
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    progress=Progress(progress_mode,heartbeat,out/'events.jsonl')
    quotes=read_quotes(quotes_file);window=window or default_window(quotes)
    amin,amax,tolerance=map(rational,(amin,amax,tolerance))
    if not 0<amin<amax or tolerance<=0:raise ValueError('Invalid variance assumptions or endpoint tolerance')
    if proposal_solver not in ('auto','cuts','full'):raise ValueError('Unknown proposal solver')
    settings=dict(version=__version__,quotes=quote_payload(quotes),window=window.json(),
        amin=str(amin),amax=str(amax),outer_levels=list(outer_levels),spline_levels=list(spline_levels),
        subdivisions=subdivisions,quadrature_cells=quadrature_cells,tolerance=str(tolerance),
        baseline_variances=[str(rational(x)) for x in baseline_variances],density_floor=str(rational(density_floor)),
        lp_time_limit=time_limit,normalization='X0=1; fixed forward/discount scenario; European calls',
        positive_denominator=bool(positive_denominator),denominator_levels=list(denominator_levels),
        expanded_models=bool(expanded_models),initial_run=str(initial_run) if initial_run else None,
        expand_marginals=bool(expand_marginals),proposal_solver=proposal_solver)
    manifest_path=out/'strong_manifest.json';manifest=dict(settings=settings,status='running',artifacts={})
    if manifest_path.exists():
        prior=json.loads(manifest_path.read_text(encoding='utf-8'))
        if resume:
            if prior['settings']!=settings:raise ValueError('Resume requires identical settings/quotes/version')
            manifest['artifacts']=prior['artifacts']
        elif not overwrite:raise ValueError('Run exists; use --resume or --overwrite')
    save_json(manifest_path,manifest);write_quotes(out/'normalized.csv',quotes)
    def stage(name,fn):
        path=out/name
        if resume and name in manifest['artifacts'] and path.exists():
            if sha256(path)!=manifest['artifacts'][name]:raise ValueError('Resume checksum mismatch: '+name)
            progress.event(name,'reused');return json.loads(path.read_text(encoding='utf-8'))
        value=fn();save_json(path,value);manifest['artifacts'][name]=sha256(path)
        save_json(manifest_path,manifest);return value
    try:
        legacy=stage('legacy_outer.json',lambda:outer_bounds(quotes,window,amin,amax,16,progress))
        if legacy['empty_class']:raise ValueError('Observations inconsistent with announced continuous class')
        lower,upper=F(legacy['lower']),F(legacy['upper'])
        floor_record=None;floor=None
        if positive_denominator:
            floor_record=stage('denominator_floor.json',lambda:denominator.build(quotes,window,amin,amax,
                denominator_levels,time_limit,progress))
            floor=F(floor_record['lower_bound'])
            if floor<F(1,2**900):raise ValueError('Positive analytical floor is below the supported binary LP scale')
        outer=[]
        for index,cells in enumerate(outer_levels):
            directory=out/f'outer_{index}'
            def calculate():
                try:
                    return occupation.bounds(quotes,window,amin,amax,directory,progress,
                        time_cells=4,strike_cells=cells,target_cells=cells,time_limit=time_limit,
                        denominator_floor=floor)
                except RuntimeError as error:
                    return dict(status='numerical_search_unresolved',reason=str(error),
                                continuous_lower=str(amin),continuous_upper=str(amax),width=str(amax-amin))
            value=stage(f'outer_{index}.json',calculate);outer.append(value)
            lower=max(lower,F(value['continuous_lower']));upper=min(upper,F(value['continuous_upper']))
        if lower>upper:raise ArithmeticError('Inconsistent exterior certificates')
        base=stage('baseline.json',lambda:inner.baseline(quotes,baseline_variances,amin,amax,progress)[0])
        region=None
        if positive_denominator and len(base['variances'])==1:
            region=stage('full_prior_region.json',lambda:time_models.full_prior_region(quotes,F(base['variances'][0]),amin,amax))
        models=[]
        if len(base['variances'])==1:
            models=stage('time_models.json',lambda:time_models.candidates(quotes,window,F(base['variances'][0]),amin,amax,
                adaptive_ramps=expanded_models))
        exact_ready=False
        if models:
            existing=endpoint_gap(lower,upper,models,tolerance)[0]
            exact_ready=F(existing['lower_endpoint_gap'])==0 and F(existing['upper_endpoint_gap'])==0
        seed_used=bool(initial_run and not exact_ready)
        if initial_run and exact_ready:progress.event('seed_models','unneeded_endpoints_already_exact')
        if seed_used:
            def seed_models():
                path=Path(initial_run)
                docs=[json.loads((path/name).read_text(encoding='utf-8')) for name in ('minimum_model.json','maximum_model.json')] if path.is_dir() else json.loads(path.read_text(encoding='utf-8'))
                if isinstance(docs,dict):docs=[docs]
                for doc in docs:
                    if F(doc['amin'])!=amin or F(doc['amax'])!=amax:raise ValueError('Seed model uses different variance priors')
                    if doc.get('model_type')=='deterministic_time_change':time_models.verify(doc,quotes,window)
                    else:inner.verify_witness(doc,quotes,window,progress,verification_cache)
                return docs
            models.extend(stage('seed_models.json',seed_models))
        records=[]
        for index,cells in enumerate(spline_levels):
            if models and endpoint_gap(lower,upper,models,tolerance)[0]['endpoints_within_tolerance_certified']:break
            reference=max((m for m in models if 'theta' in m),key=lambda m:F(m['average']['lo']),default=None)
            definition=inner.expanded_spec(window,quotes,cells,reference['definition'] if reference else None,expand_marginals) if expanded_models else inner.spec(window,quotes,cells,cells)
            endpoints=('minimum','maximum')
            if expanded_models and models:
                current=endpoint_gap(lower,upper,models,tolerance)[0]
                endpoints=tuple(label for label,key in (('minimum','lower_endpoint_gap'),('maximum','upper_endpoint_gap')) if F(current[key])>tolerance)
            value=stage(f'inner_{index}.json',lambda:inner.search(quotes,window,amin,amax,base,definition,
                subdivisions,quadrature_cells,F(density_floor),F(1),time_limit,progress,
                endpoints=endpoints,initial=reference if expanded_models else None,max_iterations=120 if expanded_models else 80,
                full_threshold=10**9 if proposal_solver=='full' else (0 if proposal_solver=='cuts' else 200)))
            models.extend(value);records.append(dict(cells=cells,models=value))
            gap,minimum,maximum=endpoint_gap(lower,upper,models,tolerance)
            progress.event('endpoint_gaps','verified',lower_gap=float(F(gap['lower_endpoint_gap'])),
                           upper_gap=float(F(gap['upper_endpoint_gap'])),tolerance=float(tolerance))
            if gap['endpoints_within_tolerance_certified']:break
        if not models:raise ValueError('No continuous witnesses constructed')
        gap,minimum,maximum=endpoint_gap(lower,upper,models,tolerance)
        result=dict(version=__version__,status='completed',settings=settings,
            outer=dict(lower=str(lower),upper=str(upper),approximate=[float(lower),float(upper)],
                       legacy=legacy,occupation_levels=outer),
            minimum_model=minimum,maximum_model=maximum,gap=gap,
            all_model_primal_exact_feasible=True,inner_levels=records)
        if floor_record is not None:result['denominator_floor']=floor_record
        if positive_denominator:
            result['window_endpoints_exact_proved']=F(gap['lower_endpoint_gap'])==0 and F(gap['upper_endpoint_gap'])==0
            result['full_prior_region']=region
        save_json(out/'minimum_model.json',minimum);save_json(out/'maximum_model.json',maximum)
        save_json(out/'result.json',result);html_report(out/'report.html',result)
        owned=['normalized.csv','legacy_outer.json','baseline.json','result.json','report.html',
               'minimum_model.json','maximum_model.json']
        if len(base['variances'])==1:owned.append('time_models.json')
        if positive_denominator:owned.append('denominator_floor.json')
        if seed_used:owned.append('seed_models.json')
        if region is not None:owned.append('full_prior_region.json')
        owned.extend(f'inner_{i}.json' for i in range(len(records)))
        owned.extend(f'outer_{i}.json' for i in range(len(outer)))
        for i,value in enumerate(outer):
            if value['status']!='numerical_search_unresolved':
                owned.extend(p.relative_to(out).as_posix() for p in (out/f'outer_{i}').glob('*') if p.suffix in ('.json','.npz'))
        manifest['artifacts']={name:sha256(out/name) for name in owned}
        manifest['status']='completed';save_json(manifest_path,manifest)
        progress.event('tighten','done',status_detail=gap['status'],output=str(out))
        return result
    except BaseException as error:
        manifest.update(status='failed',error=str(error));save_json(manifest_path,manifest);raise


def same_lp(p,q):
    return p.A.shape==q.A.shape and np.array_equal(p.A.indptr,q.A.indptr) and np.array_equal(p.A.indices,q.A.indices) and all(
        np.array_equal(a,b) for a,b in ((p.A.data,q.A.data),(p.b,q.b),(p.lo,q.lo),(p.hi,q.hi)))


def verify(directory,progress=QUIET,core_cache=None):
    if core_cache is None:core_cache={}
    out=Path(directory);manifest=json.loads((out/'strong_manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed':raise ValueError('Complete the run with --resume first')
    for name,digest in manifest['artifacts'].items():
        if sha256(out/name)!=digest:raise ValueError('Changed artifact: '+name)
    settings=manifest['settings'];quotes=read_quotes(out/'normalized.csv');w=Window.make(**settings['window'])
    amin,amax=F(settings['amin']),F(settings['amax'])
    if quote_payload(quotes)!=settings['quotes']:raise ValueError('Observation mismatch')
    legacy=outer_bounds(quotes,w,amin,amax,16,progress)
    if legacy!=json.loads((out/'legacy_outer.json').read_text(encoding='utf-8')):raise ValueError('Legacy enclosure mismatch')
    lower,upper=F(legacy['lower']),F(legacy['upper'])
    floor=None;positive_dual=F(0)
    if settings.get('positive_denominator'):
        floor_doc=json.loads((out/'denominator_floor.json').read_text(encoding='utf-8'))
        floor=denominator.verify_bundle(floor_doc,quotes,w,amin,amax,progress)
    for i,cells in enumerate(settings['outer_levels']):
        record=json.loads((out/f'outer_{i}.json').read_text(encoding='utf-8'))
        if record['status']=='numerical_search_unresolved':continue
        builder,n,d,definition=occupation.assemble(quotes,w,amin,amax,time_cells=4,
            strike_cells=cells,target_cells=cells,progress=progress,denominator_floor=floor)
        if definition!=record['definition']:raise ValueError('Continuous projection definition mismatch')
        p,f,dmin=certlp.verify(out/f'outer_{i}'/'denominator.json',progress)
        if not same_lp(p,builder.binary()) or list(f)!=list(d):raise ValueError('Outer certificate not linked to physical relaxation')
        builder.check_binary_relaxation(p)
        if str(dmin)!=record['denominator_lower']:raise ValueError('Incorrect denominator claim')
        positive_dual=max(positive_dual,dmin)
        if floor is not None:
            if dmin<floor:raise ValueError('Chosen dual discarded the proven continuous floor')
            if record.get('numerical_denominator_lower') is not None:
                pn,fn,dn=certlp.verify(out/f'outer_{i}'/'denominator_numerical.json',progress)
                if not same_lp(pn,p) or list(fn)!=list(d) or str(dn)!=record['numerical_denominator_lower']:
                    raise ValueError('Incorrect raw numerical denominator certificate')
        if dmin>0 and record['status']!='positive_denominator_normalization_ill_conditioned':
            cc=certlp.fractional_builder(p,d,dmin);cp=cc.binary();cc.check_binary_relaxation(cp)
            answers=[]
            for sign,label in ((1,'minimum'),(-1,'maximum')):
                saved,objective,value=certlp.verify(out/f'outer_{i}'/(label+'.json'),progress)
                if not same_lp(saved,cp) or objective!=[sign*x for x in n]+[F(0)]:raise ValueError('Wrong fractional projection certificate')
                answers.append(value if sign==1 else -value)
            if F(record['lower'])!=answers[0] or F(record['upper'])!=answers[1]:raise ValueError('Outer bound mismatch')
            expected_lower=max(amin,answers[0]);expected_upper=min(amax,answers[1])
        else:expected_lower,expected_upper=amin,amax
        if F(record['continuous_lower'])!=expected_lower or F(record['continuous_upper'])!=expected_upper:
            raise ValueError('Incorrect continuous bound interpretation')
        lower=max(lower,F(record['continuous_lower']));upper=min(upper,F(record['continuous_upper']))
    models=[]
    report=json.loads((out/'result.json').read_text(encoding='utf-8'))
    def check_prior(model):
        if F(model['amin'])!=amin or F(model['amax'])!=amax:raise ValueError('Witness uses a different continuous class')
    if 'time_models.json' in manifest['artifacts']:
        for model in json.loads((out/'time_models.json').read_text(encoding='utf-8')):
            check_prior(model);time_models.verify(model,quotes,w);models.append(model)
    if 'seed_models.json' in manifest['artifacts']:
        for model in json.loads((out/'seed_models.json').read_text(encoding='utf-8')):
            check_prior(model)
            if model.get('model_type')=='deterministic_time_change':time_models.verify(model,quotes,w)
            else:inner.verify_witness(model,quotes,w,progress,core_cache)
            models.append(model)
    for i,level in enumerate(report['inner_levels']):
        path=out/f'inner_{i}.json'
        if path.name not in manifest['artifacts']:raise ValueError('Untracked witness certificate')
        if level['models']!=json.loads(path.read_text(encoding='utf-8')):raise ValueError('Inconsistent witness level report')
        for model in json.loads(path.read_text(encoding='utf-8')):
            check_prior(model);inner.verify_witness(model,quotes,w,progress,core_cache);models.append(model)
    gap,minimum,maximum=endpoint_gap(lower,upper,models,F(settings['tolerance']))
    if (report['gap']!=gap or report['minimum_model']!=minimum or report['maximum_model']!=maximum or
        report['settings']!=settings or report['outer']['lower']!=str(lower) or report['outer']['upper']!=str(upper) or
        report['outer']['approximate']!=[float(lower),float(upper)]):
        raise ValueError('Reported endpoint claims differ from verified certificates')
    if floor is not None and report.get('denominator_floor')!=floor_doc:raise ValueError('Different reported denominator proof')
    if floor is not None:
        exact=F(gap['lower_endpoint_gap'])==0 and F(gap['upper_endpoint_gap'])==0
        if report.get('window_endpoints_exact_proved')!=exact:raise ValueError('Incorrect exact endpoint claim')
        region=None
        if len(settings['baseline_variances'])==1:
            region=time_models.full_prior_region(quotes,F(settings['baseline_variances'][0]),amin,amax)
            if json.loads((out/'full_prior_region.json').read_text(encoding='utf-8'))!=region:raise ValueError('Incorrect continuum-region certificate')
        if report.get('full_prior_region')!=region:raise ValueError('Different reported continuum-region certificate')
    if (minimum!=json.loads((out/'minimum_model.json').read_text(encoding='utf-8')) or
        maximum!=json.loads((out/'maximum_model.json').read_text(encoding='utf-8'))):raise ValueError('Model export differs from verified model')
    known_floor=floor if floor is not None else positive_dual
    return dict(verified=True,continuous_projection_recomputed=True,continuous_models_verified=len(models),
                denominator_positive_proved=known_floor>0,denominator_lower=str(known_floor) if known_floor>0 else None,
                primal_exact_feasible=True,endpoint_gaps=gap,optimizer_restarted=False)
