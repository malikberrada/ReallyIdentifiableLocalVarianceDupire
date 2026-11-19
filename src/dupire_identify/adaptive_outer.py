"""Retain the best exact outer bounds across physical mesh refinements."""
from fractions import Fraction as F
from pathlib import Path
import json
from . import signed_outer
from .io import save_json
from .progress import QUIET
from .display import approximate


def summary(family,amin,amax,records):
    lower,upper=F(amin),F(amax);empty=False
    for r in records:
        if r['kind']=='verified_outer':
            lower=max(lower,F(r['lower']));upper=min(upper,F(r['upper']));empty|=r['empty_class_certified']
    if lower>upper:raise ArithmeticError('Contradictory verified mesh refinements')
    return dict(family=family.json(),amin=str(amin),amax=str(amax),records=records,lower=str(lower),upper=str(upper),
        empty_class_certified=bool(empty),uniform_convergence_proved=False)


def run(quotes,family,amin,amax,directory,strike_cells=6,target_cells=6,steps=6,time_limit=30,refinements=1,progress=QUIET):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    if refinements<1:raise ValueError('Positive physical refinement budget required')
    records=[]
    for level in range(refinements):
        folder=f'level_{level:03d}';settings=dict(strike_cells=strike_cells*2**level,target_cells=target_cells*2**level,
            steps=steps,time_limit=time_limit,time_cells=4*2**min(level,1))
        if (out/folder/'outer.json').exists():
            record=signed_outer.verify(quotes,family,amin,amax,out/folder,progress)
        else:
            try:record=signed_outer.run(quotes,family,amin,amax,out/folder,progress=progress,**settings)
            except (ValueError,OverflowError,RuntimeError) as error:
                records.append(dict(kind='numeric_search_unresolved',level=level,reason=str(error)))
                progress.event('physical_mesh_refinement','unresolved',level=level,reason=str(error));continue
        records.append(dict(kind='verified_outer',level=level,directory=folder,
            lower=record['lower'],upper=record['upper'],empty_class_certified=record['empty_class_certified']))
        progress.event('physical_mesh_refinement','done',done=level+1,total=refinements,
            lower=approximate(F(record['lower'])),upper=approximate(F(record['upper'])))
        if record['empty_class_certified']:break
    result=summary(family,amin,amax,records);save_json(out/'adaptive.json',result);return result


def verify(quotes,family,amin,amax,directory,progress=QUIET):
    out=Path(directory);result=json.loads((out/'adaptive.json').read_text(encoding='utf-8'));records=[]
    for i,r in enumerate(result['records']):
        if r['level']!=i:raise ValueError('Missing physical refinement record')
        if r['kind']=='numeric_search_unresolved':
            if not isinstance(r['reason'],str):raise ValueError('Invalid numeric search failure')
            records.append(r);continue
        folder=f'level_{i:03d}'
        if r['kind']!='verified_outer' or r['directory']!=folder:raise ValueError('Invalid refinement certificate path')
        checked=signed_outer.verify(quotes,family,amin,amax,out/folder,progress)
        expected=dict(kind='verified_outer',level=i,directory=folder,lower=checked['lower'],upper=checked['upper'],empty_class_certified=checked['empty_class_certified'])
        if r!=expected:raise ValueError('Incorrect physical refinement summary')
        records.append(expected)
    if result!=summary(family,amin,amax,records):raise ValueError('Incorrect combined outer interval')
    return result
