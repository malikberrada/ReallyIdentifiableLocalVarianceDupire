"""Repair completed 0.5.0 runs in a separate output, preserving old proofs."""
from pathlib import Path
import json
import shutil
from . import exhaustion
from .progress import Progress


def run(source,output,progress_mode='plain',heartbeat=2,backend='auto'):
    source=Path(source).resolve();out=Path(output).resolve()
    if out==source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('Recovery needs a separate, non-nested output directory')
    if out.exists() and any(out.iterdir()):raise ValueError('Recovery output must be empty')
    m=json.loads((source/'exhaustion_manifest.json').read_text(encoding='utf-8'))
    if m['status']!='completed':raise ValueError('Recover a completed prefix; use --resume for incomplete new-version runs')
    # Validate hashes before copying, without changing the source archive.
    from .io import sha256
    for n,h in m['artifacts'].items():
        p=Path(n)
        if p.is_absolute() or '..' in p.parts or sha256(source/p)!=h:raise ValueError('Changed source artifact: '+n)
    shutil.copytree(source,out,dirs_exist_ok=True)
    for shell in out.glob('shell_*'):
        for n in ('general_manifest.json','frontier.json','frontier.html','partial.json'):
            (shell/n).unlink(missing_ok=True)
        for node in shell.glob('nodes/*/node.json'):node.unlink()
    for n in ('exhaustion_manifest.json','exhaustion.json','partial.json'):(out/n).unlink(missing_ok=True)
    settings=m['settings'];options=dict(settings['options'])
    options.update(fit_models=0,retry_unresolved=True,backend=backend)
    indices=settings['indices'];quotes=out/f'shell_{indices[0]:04d}'/'normalized.csv'
    progress=Progress(progress_mode,heartbeat,out/'recovery_events.jsonl')
    progress.event('recovery','start',shells=len(indices),source_preserved=True,
        note='Reuse and reverify completed duals; retry numerical failures only')
    result=exhaustion.run(quotes,out,indices,(),progress_mode,heartbeat,False,**options)
    progress.event('recovery','done',all_requested_shells_solved=result['all_requested_shells_solved'])
    return result
