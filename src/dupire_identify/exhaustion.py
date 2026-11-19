"""Countable coverage scheme for ALL positive kernel parameters.

u=t-hT, v=k-hK, hT,hK >0. The nested boxes [2^-j,2^j]^4 exhaust this
space. A finite prefix certifies only the requested boxes, never infinity.
"""
from pathlib import Path
import json
from . import general_frontier
from .families import shell
from .io import save_json,sha256,read_quotes,quote_payload
from .progress import Progress,QUIET


def run(quotes,output,indices=(0,1),model_runs=(),progress_mode='plain',heartbeat=2,resume=False,**options):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    indices=list(indices)
    if not indices or len(set(indices))!=len(indices) or any(i<0 for i in indices):raise ValueError('Distinct nonnegative shell indices required')
    progress=Progress(progress_mode,heartbeat,out/'events.jsonl');manifest=out/'exhaustion_manifest.json'
    settings=dict(indices=indices,quotes=quote_payload(read_quotes(quotes)),options=options)
    if manifest.exists():
        old=json.loads(manifest.read_text(encoding='utf-8'))
        if not resume or old['settings']!=settings:raise ValueError('Use --resume with identical shell settings')
        if old['status']=='completed':verify(out,progress);return json.loads((out/'exhaustion.json').read_text(encoding='utf-8'))
    save_json(manifest,dict(status='running',settings=settings));records=[];sources=list(model_runs)
    for n,j in enumerate(indices):
        folder=f'shell_{j:04d}';progress.event('positive_parameter_shell','start',done=n,total=len(indices),index=j)
        result=general_frontier.run(quotes,out/folder,shell(j),model_runs=sources,progress_mode=progress_mode,heartbeat=heartbeat,
            resume=resume and (out/folder/'general_manifest.json').exists(),**options)
        records.append(dict(index=j,directory=folder,status=result['status'],requested_domain_solved=result['all_parameters_in_requested_domain_solved']))
        if not sources:
            bank=json.loads((out/folder/'bank/index.json').read_text(encoding='utf-8'))
            sources=[str(out/folder/e['directory']) for e in bank]
        save_json(out/'partial.json',dict(settings=settings,completed=records))
        progress.event('positive_parameter_shell','done',done=n+1,total=len(indices),index=j,status_detail=result['status'])
    result=dict(settings=settings,shells=records,all_requested_shells_solved=all(r['requested_domain_solved'] for r in records),
        mathematical_scheme_covers_all_positive_parameters=True,
        scheme='u=t-hT>0, v=k-hK>0; union over ALL j>=0 of [2^-j,2^j]^4',
        all_positive_parameters_solved=False,all_smiles_solved=False,finite_prefix_only=True)
    save_json(out/'exhaustion.json',result)
    names=['exhaustion.json']
    for record in records:
        folder=record['directory'];m=json.loads((out/folder/'general_manifest.json').read_text(encoding='utf-8'))
        names.append(folder+'/general_manifest.json');names.extend(folder+'/'+n for n in m['artifacts'])
    save_json(manifest,dict(status='completed',settings=settings,artifacts={n:sha256(out/n) for n in names}))
    return result


def verify(output,progress=QUIET):
    out=Path(output);m=json.loads((out/'exhaustion_manifest.json').read_text(encoding='utf-8'))
    if m['status']!='completed':raise ValueError('Incomplete shell prefix')
    for n,h in m['artifacts'].items():
        p=Path(n)
        if p.is_absolute() or '..' in p.parts or sha256(out/p)!=h:raise ValueError('Changed exhaustion artifact')
    result=json.loads((out/'exhaustion.json').read_text(encoding='utf-8'));settings=m['settings'];records=[]
    for j,record in zip(settings['indices'],result['shells']):
        folder=f'shell_{j:04d}'
        if record['index']!=j or record['directory']!=folder:raise ValueError('Missing or changed parameter shell')
        checked=general_frontier.verify(out/folder,progress)
        sub=json.loads((out/folder/'frontier.json').read_text(encoding='utf-8'))
        if sub['settings']['family']!=shell(j).json() or sub['settings']['quotes']!=settings['quotes']:
            raise ValueError('Shell is not linked to the declared scheme or input smile')
        for k,v in settings['options'].items():
            if k in sub['settings']:
                if k in ('amin','amax','tolerance'):
                    from fractions import Fraction as F
                    if F(sub['settings'][k])!=F(v):raise ValueError('Shell uses different rational settings')
                elif sub['settings'][k]!=v:raise ValueError('Shell uses different settings')
        records.append(dict(index=j,directory=folder,status=sub['status'],requested_domain_solved=checked['all_parameters_in_requested_domain_solved']))
    expected=dict(settings=settings,shells=records,all_requested_shells_solved=all(r['requested_domain_solved'] for r in records),
        mathematical_scheme_covers_all_positive_parameters=True,
        scheme='u=t-hT>0, v=k-hK>0; union over ALL j>=0 of [2^-j,2^j]^4',
        all_positive_parameters_solved=False,all_smiles_solved=False,finite_prefix_only=True)
    if len(result['shells'])!=len(settings['indices']) or result!=expected:raise ValueError('False unbounded completion claim')
    return dict(verified=True,all_requested_shells_solved=expected['all_requested_shells_solved'],
        all_positive_parameters_solved=False,all_smiles_solved=False,finite_prefix_only=True,
        optimizer_restarted=False,pde_solver_restarted=False)
