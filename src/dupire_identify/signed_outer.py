"""Family-wide continuous bounds from unnormalised signed LP objectives.

Every saved bound is an exact dual inequality. Numeric LP primal solutions
never become feasible diffusions. Phase-I can certify an empty model class.
"""
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from . import occupation,certlp,denominator
from .families import Family
from .native_solver import WarmLP,optimize
from .io import save_json
from .progress import QUIET
from .tightening import same_lp
from .display import approximate


def phase_one(p):
    if np.any(p.lo!=0):raise ValueError('Phase-I assumes nonnegative variables')
    b=certlp.RationalLP()
    for u in p.hi:b.var(F(float(u)))
    slack=b.var(1+max([F(0)]+[-F(float(v)) for v in p.b]),'uniform_violation')
    for i in range(p.A.shape[0]):
        row={int(p.A.indices[j]):F(float(p.A.data[j])) for j in range(p.A.indptr[i],p.A.indptr[i+1])}
        row[slack]=-1;b.row(row,F(float(p.b[i])))
    f=[F(0)]*len(b.upper);f[-1]=1
    return b,f


def finite_dual(res,rows):
    if res is not None and getattr(res,'dual_finite',False):return res
    return SimpleNamespace(ineqlin=SimpleNamespace(marginals=np.zeros(rows)))


def saved_certificate(path,p,objective,progress):
    """Re-use a completed dual only after checking its exact physical link."""
    if not Path(path).exists():return None
    stored,stored_objective,value=certlp.verify(path,progress)
    if not same_lp(stored,p) or stored_objective!=objective:
        raise ValueError('Saved dual belongs to another physical relaxation')
    progress.event('continuous_dual','reused',certificate=Path(path).name)
    return value


def continuous_floor(quotes,family,amin,amax,levels=(4,),time_limit=30,progress=QUIET):
    common=family.common_kernel()
    if common is None:return None,F(0)
    ref,factor=common
    try:
        bundle=denominator.build(quotes,ref,amin,amax,levels,time_limit,progress)
    except ValueError as exc:
        return dict(status='no_numeric_floor_within_arithmetic_budget',reason=str(exc)),F(0)
    value=factor*F(bundle['lower_bound'])
    # The physical LP can use a representable dyadic lower floor. A weaker
    # floor remains a valid statement; no negative value is flipped by convention.
    floor=min(F(1,2),denominator.dyadic_floor(value))
    row_floor=floor if floor>=F(1,2**900) else None
    return dict(status='common_kernel_floor_verified',reference=ref.json(),factor=str(factor),
                bundle=bundle,lower=str(floor),lp_row_floor=str(row_floor) if row_floor else None),floor


def verify_floor(record,quotes,family,amin,amax,progress=QUIET):
    floor=F(0)
    if record and record.get('status')=='common_kernel_floor_verified':
        common=family.common_kernel()
        if common is None:raise ValueError('No common kernel for claimed floor')
        ref,factor=common
        if record['reference']!=ref.json() or F(record['factor'])!=factor:raise ValueError('Incorrect kernel domination')
        value=denominator.verify_bundle(record['bundle'],quotes,ref,F(amin),F(amax),progress)
        floor=min(F(1,2),denominator.dyadic_floor(factor*value))
        row=floor if floor>=F(1,2**900) else None
        if record['lower']!=str(floor) or record['lp_row_floor']!=(str(row) if row else None):raise ValueError('Incorrect family floor')
    elif record is not None and record.get('status')!='no_numeric_floor_within_arithmetic_budget':
        raise ValueError('Unknown floor record')
    return floor


def assemble(quotes,family,amin,amax,settings,floor_record,progress=QUIET):
    row_floor=F(floor_record['lp_row_floor']) if floor_record and floor_record.get('lp_row_floor') else None
    return occupation.assemble(quotes,family.cover_window(),amin,amax,
        time_cells=settings['time_cells'],strike_cells=settings['strike_cells'],
        target_cells=settings['target_cells'],time_degree=settings['time_degree'],
        comparison_cuts=settings['comparison_cuts'],denominator_floor=row_floor,
        window_family=family,progress=progress)


def run(quotes,family,amin,amax,directory,strike_cells=8,target_cells=8,time_cells=4,
        steps=8,time_limit=60,denominator_levels=(4,),comparison_cuts=True,progress=QUIET):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    amin,amax=F(amin),F(amax)
    settings=dict(strike_cells=int(strike_cells),target_cells=int(target_cells),time_cells=int(time_cells),
        time_degree=1,comparison_cuts=bool(comparison_cuts),steps=int(steps),time_limit=float(time_limit))
    if not 0<amin<amax or steps<0 or time_limit<=0:raise ValueError('Invalid outer-bound settings')
    if (out/'floor.json').exists():
        floor_record=json.loads((out/'floor.json').read_text(encoding='utf-8'))
        floor=verify_floor(floor_record,quotes,family,amin,amax,progress)
    else:floor_record,floor=continuous_floor(quotes,family,amin,amax,denominator_levels,min(time_limit,30),progress)
    save_json(out/'floor.json',floor_record)
    with progress.phase('general_continuous_relaxation'):
        builder,n,d,definition=assemble(quotes,family,amin,amax,settings,floor_record,progress)
    p=builder.binary();builder.check_binary_relaxation(p)
    phase,f=phase_one(p);pp=phase.binary();phase.check_binary_relaxation(pp)
    violation=saved_certificate(out/'phase_one.json',pp,f,progress)
    if violation is None:
        r=optimize(list(map(float,f)),pp.A,pp.b,pp.lo,pp.hi,time_limit,progress=progress,label='model_class_phase_I')
        violation=certlp.certificate(out/'phase_one.json',pp,f,finite_dual(r,pp.A.shape[0]),progress)
    result=dict(family=family.json(),amin=str(amin),amax=str(amax),settings=settings,definition=definition,
        floor=floor_record,phase_one_lower=str(violation),empty_class_certified=violation>0,
        lower=str(amin),upper=str(amax),denominator_lower=str(floor),tests=[],
        scope='Every continuous model compatible with the quotes, for every window in this family',
        uniform_convergence_proved=False)
    if violation>0:
        result['status']='empty_continuous_class_certified';save_json(out/'outer.json',result);return result
    engine=WarmLP(np.asarray(list(map(float,d))),p.A,p.b,p.lo,p.hi,solver='simplex')
    dlp=saved_certificate(out/'denominator.json',p,d,progress)
    if dlp is None:
        r=engine.solve(list(map(float,d)),time_limit,progress,'family_denominator_search')
        dlp=certlp.certificate(out/'denominator.json',p,d,finite_dual(r,p.A.shape[0]),progress)
    positive=max(floor,dlp);result.update(denominator_lp_lower=str(dlp),denominator_lower=str(positive))
    if positive<=0:
        result['status']='outer_prior_only_no_numeric_denominator';save_json(out/'outer.json',result);return result
    lower,upper=amin,amax
    for side in ('minimum','maximum'):
        a,b=amin,amax
        for i in range(steps):
            center=(a+b)/2
            objective=[x-center*y for x,y in zip(n,d)] if side=='minimum' else [center*y-x for x,y in zip(n,d)]
            vector=np.asarray(list(map(float,objective)))
            scale=max(1.,float(np.max(abs(vector))))
            name=f'{side}_{i:03d}.json'
            value=saved_certificate(out/name,p,objective,progress)
            if value is None:
                r=engine.solve(vector/scale,time_limit,progress,'signed_'+side)
                value=certlp.certificate(out/name,p,objective,finite_dual(r,p.A.shape[0]),progress)
            correction=min(F(0),value)/positive
            bound=center+correction if side=='minimum' else center-correction
            if side=='minimum':lower=max(lower,bound)
            else:upper=min(upper,bound)
            # A proposal bracket for the RELAXATION. It is never reported as
            # an inner bound for the physical diffusion class.
            if (side=='minimum')==(value>=0):a=center
            else:b=center
            result['tests'].append(dict(side=side,r=str(center),file=name,dual_lower=str(value),physical_bound=str(bound)))
            progress.event('signed_fractional_search','running',done=i+1,total=steps,side=side,
                bound=approximate(bound),preview_out_of_range=approximate(bound) is None)
    if lower>upper:raise ArithmeticError('Contradictory signed bounds')
    result.update(status='continuous_family_outer_verified',lower=str(lower),upper=str(upper))
    save_json(out/'outer.json',result);return result


def verify(quotes,family,amin,amax,directory,progress=QUIET):
    out=Path(directory);result=json.loads((out/'outer.json').read_text(encoding='utf-8'))
    if result['family']!=family.json() or F(result['amin'])!=F(amin) or F(result['amax'])!=F(amax):raise ValueError('Different outer problem')
    record=json.loads((out/'floor.json').read_text(encoding='utf-8'))
    floor=verify_floor(record,quotes,family,amin,amax,progress)
    if result['floor']!=record:raise ValueError('Changed floor summary')
    builder,n,d,definition=assemble(quotes,family,F(amin),F(amax),result['settings'],record,progress)
    if definition!=result['definition']:raise ValueError('Different family relaxation')
    p=builder.binary();builder.check_binary_relaxation(p)
    phase,f=phase_one(p);pp=phase.binary();phase.check_binary_relaxation(pp)
    saved,obj,violation=certlp.verify(out/'phase_one.json',progress)
    if not same_lp(saved,pp) or obj!=f or str(violation)!=result['phase_one_lower']:
        raise ValueError('Phase-I not linked to the physical relaxation')
    empty=violation>0
    if empty!=result['empty_class_certified']:raise ValueError('Incorrect feasibility claim')
    lower,upper=F(amin),F(amax)
    if not empty:
        saved,obj,dlp=certlp.verify(out/'denominator.json',progress)
        if not same_lp(saved,p) or obj!=d or str(dlp)!=result['denominator_lp_lower']:raise ValueError('Wrong denominator dual')
        positive=max(floor,dlp)
        if str(positive)!=result['denominator_lower']:raise ValueError('Incorrect denominator lower bound')
        expected=[(s,i) for s in ('minimum','maximum') for i in range(result['settings']['steps'])] if positive>0 else []
        if len(result['tests'])!=len(expected):raise ValueError('Missing signed objective certificates')
        a,b=F(amin),F(amax);previous=None
        for test,(side,i) in zip(result['tests'],expected):
            if side!=previous:a,b=F(amin),F(amax);previous=side
            center=(a+b)/2;name=f'{side}_{i:03d}.json'
            if test['side']!=side or test['file']!=name or F(test['r'])!=center:raise ValueError('Changed signed search')
            objective=[x-center*y for x,y in zip(n,d)] if side=='minimum' else [center*y-x for x,y in zip(n,d)]
            saved,obj,value=certlp.verify(out/name,progress)
            if not same_lp(saved,p) or obj!=objective or str(value)!=test['dual_lower']:raise ValueError('Unlinked signed certificate')
            bound=center+min(F(0),value)/positive if side=='minimum' else center-min(F(0),value)/positive
            if str(bound)!=test['physical_bound']:raise ValueError('Incorrect ratio interpretation')
            if side=='minimum':lower=max(lower,bound)
            else:upper=min(upper,bound)
            if (side=='minimum')==(value>=0):a=center
            else:b=center
    else:
        if result['tests'] or result['denominator_lower']!=str(floor):raise ValueError('Claim after an empty-class proof')
    expected_status=('empty_continuous_class_certified' if empty else 'continuous_family_outer_verified' if F(result['denominator_lower'])>0 else 'outer_prior_only_no_numeric_denominator')
    if result['lower']!=str(lower) or result['upper']!=str(upper) or result['status']!=expected_status or result['uniform_convergence_proved']:
        raise ValueError('Incorrect family outer summary')
    return result
