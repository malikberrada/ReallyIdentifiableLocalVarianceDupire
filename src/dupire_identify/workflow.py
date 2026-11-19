from fractions import Fraction as F
import csv
import html
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import scipy
from . import __version__
from .io import save_json,sha256,read_quotes,write_quotes,Quote,rational,payload_quotes
from .continuous import Window,nested_outer,outer_bounds
from .witness import witness_pair,choose_gap
from .lp import build_polytope,identify_discrete,verify_certificate
from .backend import Backend
from .progress import Progress,QUIET


def synthetic_quotes(path,spread="1/500",a0="1/25",backend="numpy",progress=QUIET):
    a0=rational(a0);half=rational(spread)
    if half<=0:raise ValueError("Positive spread required")
    dates=[F(1,4),F(3,4),F(1)]
    ks=[F(x,100) for x in (72,83,91,97,100,104,113,129,147)]
    engine=Backend(backend,progress=progress)
    prices=engine.black([float(k) for k in ks],[float(a0*t) for t in dates])
    quotes=[]
    for i,t in enumerate(dates):
        for j,k in enumerate(ks):
            mid=F(format(max(0,prices[i,j]),".14f"))
            quotes.append(Quote(f"synthetic_{i}_{j}",t,k,max(F(0),1-k,mid-half),min(F(1),mid+half)))
    write_quotes(path,quotes)
    return quotes


def default_window(quotes):
    left,right=choose_gap(quotes);gap=right-left
    return Window.make(left+gap/4,1,gap/8,"1/10")


def report_html(directory,report):
    def esc(x):return html.escape(str(x))
    rows=[]
    for title,result in (("P1 continu",report.get('continuous')),("P1 grille",report.get('discrete')),
                          ("Temoins P2",report.get('witness'))):
        if not result:continue
        value=result.get('approximate',result.get('status'))
        rows.append(f"<tr><td>{title}</td><td>{esc(result.get('scope'))}</td><td>{esc(value)}</td></tr>")
    body=f"""<!doctype html><html lang="fr"><meta charset="utf-8"><title>Dupire Identify</title>
<style>body{{font:16px system-ui;max-width:1080px;margin:40px auto;padding:24px;color:#142435;background:#f5f8fc}}
table{{border-collapse:collapse;width:100%;background:white}}td,th{{border:1px solid #ccd5df;padding:14px;text-align:left}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:24px}}h1{{color:#143b65}}
.note{{border-left:4px solid #ce8d20;padding:16px;background:#fff7e6}}</style>
<h1>Resolution identifiable de la variance locale</h1>
<p>Moyenne ponderee par la densite risque neutre et par k<sup>2</sup>. Variance en unites annuelles.</p>
<p class="note">Les bornes de grille ne sont pas des bornes de toute la classe continue.
Un intervalle continu large est un resultat valide. Aucune optimalite de la frontiere n'est affirmee.</p>
<table><tr><th>Calcul</th><th>Portee exacte</th><th>Resultat</th></tr>{''.join(rows)}</table>
<h2>Decision</h2><pre>{esc(json.dumps(report.get('decision'),indent=2,ensure_ascii=False))}</pre>
<h2>Configuration et resultats</h2><pre>{esc(json.dumps(report,indent=2,ensure_ascii=False))}</pre>
</html>"""
    (Path(directory)/'report.html').write_text(body,encoding='utf-8')


def run_identification(quotes_file,output,window=None,amin="1/400",amax="1/4",
                       levels=(8,16),nt=16,nk=35,epsilon="1/100",a0="1/25",
                       witness_epsilon="1/50",backend="auto",threads=None,time_limit=120,
                       discrete=True,progress_mode="plain",heartbeat=2,resume=False,overwrite=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    progress=Progress(progress_mode,heartbeat,out/'events.jsonl')
    quotes=read_quotes(quotes_file)
    window=window or default_window(quotes)
    amin,amax,epsilon=map(rational,(amin,amax,epsilon))
    if not 0<amin<=amax or epsilon<=0 or not levels or any(int(x)<2 for x in levels):
        raise ValueError("Invalid assumptions, precision target or refinement levels")
    settings=dict(version=__version__,quotes_sha256=sha256(quotes_file),window=window.json(),
                  amin=str(amin),amax=str(amax),levels=list(levels),nt=nt,nk=nk,
                  epsilon=str(epsilon),a0=str(rational(a0)),witness_epsilon=str(rational(witness_epsilon)),
                  discrete=discrete)
    manifest_path=out/'manifest.json'
    previous=None
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text(encoding='utf-8'))
        if resume and previous['settings']!=settings:
            raise ValueError("--resume requires identical input hash, version and settings")
        if not resume and not overwrite:
            raise ValueError("Output already has a run; use --resume or --overwrite")
    manifest=dict(settings=settings,status="running",artifacts={},environment=dict(
        python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()))
    if resume:manifest['artifacts']=previous.get('artifacts',{})
    save_json(manifest_path,manifest)
    def checkpoint(name,fn):
        path=out/name
        if resume and name in manifest['artifacts'] and path.exists():
            if sha256(path)!=manifest['artifacts'][name]:raise ValueError(f"Resume checksum mismatch: {name}")
            progress.event(name,"reused_verified_checksum")
            return json.loads(path.read_text(encoding='utf-8'))
        value=fn();save_json(path,value)
        manifest['artifacts'][name]=sha256(path);save_json(manifest_path,manifest)
        return value
    try:
        engine=Backend(backend,threads,progress)
        write_quotes(out/'normalized.csv',quotes)
        cont=checkpoint('continuous.json',lambda:nested_outer(quotes,window,amin,amax,levels,progress))
        if cont['empty_class']:
            result=dict(version=__version__,status="empty_continuous_class",continuous=cont,
                        discrete=None,witness=None,backend=engine.name,settings=settings,
                        decision=dict(status='empty_class',attainable_identification_frontier_proved=False))
            save_json(out/'result.json',result);report_html(out,result)
            for path in (out/'normalized.csv',out/'result.json',out/'report.html'):
                manifest['artifacts'][path.name]=sha256(path)
            manifest['status']='empty_class'
            save_json(manifest_path,manifest);return result
        def witnesses():
            try:
                return witness_pair(quotes,a0,witness_epsilon,amin,amax,window=window,progress=progress)
            except ValueError as e:
                return dict(scope="continuous_analytic_deterministic_variance_subclass",certified=False,
                            status="witness_not_constructed",reason=str(e))
        witness=checkpoint('witness.json',witnesses)
        grid=None
        if discrete:
            def finite():
                p=build_polytope(quotes,amin,amax,nt,nk,progress=progress)
                return identify_discrete(p,window,out/'lp',progress,time_limit)
            # A finite LP can be infeasible even if continuous witnesses exist.
            # Keep the verified continuous outputs and flag this independently.
            def safe_finite():
                try:return finite()
                except (RuntimeError,ValueError) as e:
                    progress.event('finite_grid','unresolved',error=str(e))
                    return dict(scope="finite_grid_only",status="unresolved",reason=str(e),continuous_bound=False)
            grid=checkpoint('discrete.json',safe_finite)
        width=F(cont['width'])
        gap=F(witness.get('separation_lower','0')) if witness.get('certified') else F(0)
        if gap>width:
            raise ArithmeticError("Witness separation exceeds continuous outer interval: internal inconsistency")
        if width<=2*epsilon:decision="precision_certified_under_assumptions"
        elif gap>2*epsilon:decision="requested_precision_impossible_by_continuous_witnesses"
        else:decision="unresolved"
        result=dict(version=__version__,status="completed",scope="research_foundation",
                    continuous=cont,discrete=grid,witness=witness,backend=engine.name,
                    decision=dict(status=decision,requested_max_error=str(epsilon),
                                  continuous_outer_width=str(width),witnessed_separation_lower=str(gap),
                                  attainable_identification_frontier_proved=False),settings=settings)
        save_json(out/'result.json',result);report_html(out,result)
        owned=[out/'result.json',out/'report.html',out/'normalized.csv']
        if Path(quotes_file).resolve().parent==out.resolve():owned.append(Path(quotes_file))
        if grid and grid.get('status')=='finite_lp_bounds_verified':
            owned.extend((out/'lp').glob('*.json'));owned.extend((out/'lp').glob('*.npz'))
        for path in owned:
            manifest['artifacts'][path.relative_to(out).as_posix()]=sha256(path)
        manifest['status']='completed';save_json(manifest_path,manifest)
        progress.event('run','done',output=str(out),decision=decision)
        return result
    except BaseException as e:
        manifest.update(status="failed",error=str(e));save_json(manifest_path,manifest)
        progress.event('run','failed',error=str(e))
        raise


def verify_run(directory,progress=QUIET):
    out=Path(directory)
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] not in ('completed','empty_class'):
        raise ValueError('Run incomplete; finish it with --resume before verifying the report')
    for name,checksum in manifest['artifacts'].items():
        if sha256(out/name)!=checksum:raise ValueError(f"Run checksum mismatch: {name}")
    settings=manifest['settings'];quotes=read_quotes(out/'normalized.csv')
    w=Window.make(**settings['window'])
    previous=json.loads((out/'continuous.json').read_text(encoding='utf-8'))
    recomputed=nested_outer(quotes,w,settings['amin'],settings['amax'],settings['levels'],progress)
    if recomputed!=previous:raise ValueError("Continuous enclosure recomputation mismatch")
    if previous['empty_class']:
        result=json.loads((out/'result.json').read_text(encoding='utf-8'))
        if (result['continuous']!=previous or result['settings']!=settings or
            result['status']!='empty_continuous_class' or result['decision']!=dict(
                status='empty_class',attainable_identification_frontier_proved=False)):
            raise ValueError('Incorrect empty-class report')
        return dict(verified=True,empty_class=True,continuous_enclosures_recomputed=True,
                    reported_decision_verified=True,attainable_frontier_proved=False)
    witness=json.loads((out/'witness.json').read_text(encoding='utf-8')) if (out/'witness.json').exists() else None
    if witness and witness.get('certified'):
        check=witness_pair(quotes,witness['a0'],witness['epsilon'],witness['amin'],witness['amax'],
                           witness['left'],witness['right'],w,progress)
        if check!=witness:raise ValueError("Continuous witness recomputation mismatch")
    certs=[]
    for name in sorted(manifest['artifacts']):
        if name.startswith('lp/') and name.endswith('.json'):
            certs.append(verify_certificate(out/name,progress))
    result=json.loads((out/'result.json').read_text(encoding='utf-8'))
    if result['continuous']!=previous or result['witness']!=witness or result['settings']!=settings:
        raise ValueError('Result report differs from its verified inputs')
    width=F(previous['width']);eps=F(settings['epsilon'])
    gap=F(witness.get('separation_lower','0')) if witness and witness.get('certified') else F(0)
    decision=('precision_certified_under_assumptions' if width<=2*eps else
              'requested_precision_impossible_by_continuous_witnesses' if gap>2*eps else 'unresolved')
    expected=dict(status=decision,requested_max_error=str(eps),continuous_outer_width=str(width),
                  witnessed_separation_lower=str(gap),attainable_identification_frontier_proved=False)
    if result['decision']!=expected:raise ValueError('Incorrect reported identification decision')
    finite=result.get('discrete') or {}
    if finite.get('status')=='finite_lp_bounds_verified':
        low=json.loads((out/'lp'/'fractional_min.json').read_text(encoding='utf-8'))
        high=json.loads((out/'lp'/'fractional_max.json').read_text(encoding='utf-8'))
        if F(finite['lower'])!=F(low['lower_bound']) or F(finite['upper'])!=-F(high['lower_bound']):
            raise ValueError('Reported finite bounds differ from their certificates')
    return dict(verified=True,continuous_enclosures_recomputed=True,
                continuous_witnesses_recomputed=bool(witness and witness.get('certified')),
                finite_lp_certificates=certs,reported_decision_verified=True,attainable_frontier_proved=False)


def frontier(quotes_file,output,t,k,hts,hks,amin="1/400",amax="1/4",epsilon="1/100",
             cells=16,progress=QUIET):
    quotes=read_quotes(quotes_file);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    results=[];total=len(hts)*len(hks)
    for ht in hts:
        for hk in hks:
            w=Window.make(t,k,ht,hk)
            with progress.phase('frontier_window',window=w.json(),index=len(results)+1,total_windows=total):
                r=outer_bounds(quotes,w,amin,amax,cells,progress)
            status="empty_class" if r['empty_class'] else (
                "certified_precision" if F(r['width'])<=2*rational(epsilon) else "unresolved")
            results.append(dict(window=w.json(),status=status,continuous=r))
            save_json(out/'frontier.json',results)
            progress.event('frontier_windows','running',done=len(results),total=total)
    with (out/'frontier.csv').open('w',encoding='utf-8',newline='') as f:
        wr=csv.writer(f);wr.writerow(['t','k','ht','hk','lower_variance','upper_variance','width','status'])
        for row in results:
            wr.writerow([*row['window'].values(),row['continuous'].get('lower',''),row['continuous'].get('upper',''),
                         row['continuous'].get('width',''),row['status']])
    return results
