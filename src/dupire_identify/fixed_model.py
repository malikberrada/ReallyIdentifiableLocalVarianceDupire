"""Verified diffusion bank and weak observables on arbitrary window families."""
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
from math import comb
import json
import numpy as np
from . import __version__,plateau,binary_intervals as bi,exact_endpoint,denominator
from .binary_intervals import B
from .continuous import Window
from .interval import I,black_call,exp_point
from .polynomial import bump,deriv,mul,integrate,bernstein_range
from .io import read_quotes,write_quotes,quote_payload,save_json,sha256
from .progress import QUIET,Progress

FILES=('normalized.csv','diffusion.json','surface.npz','mesh.json','pricing.json')
_VERIFIED_MODELS={}


def proof_key(directory):
    """Content identity; actual artifact bytes are hashed on every cache use."""
    out=Path(directory)
    name='model_manifest.json' if (out/'model_manifest.json').exists() else 'exact_manifest.json'
    m=json.loads((out/name).read_text(encoding='utf-8'));items=[]
    if m['status']!='completed':raise ValueError('Incomplete fixed-model proof')
    for filename,digest in sorted(m['artifacts'].items()):
        p=Path(filename)
        if p.is_absolute() or '..' in p.parts or sha256(out/p)!=digest:
            raise ValueError('Changed fixed-model artifact: '+filename)
        items.append((filename,digest))
    return name,sha256(out/name),tuple(items)


def verify_cached(directory,progress=QUIET):
    """Search-only cache. Public verify() always recomputes the full proof."""
    key=proof_key(directory)
    if key not in _VERIFIED_MODELS:_VERIFIED_MODELS[key]=verify(directory,progress)
    else:progress.event('fixed_model_proof','reused_identical_bytes')
    return dict(_VERIFIED_MODELS[key])


def run(config,quotes_file,directory,spatial_cells=128,pulse_cells=16,time_steps=512,refinements=2,
        progress_mode='plain',heartbeat=2,storage_stride=4):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    if (out/'model_manifest.json').exists():raise ValueError('Fixed model exists; verify or use another output')
    q=read_quotes(quotes_file);plateau.validate_config(config);progress=Progress(progress_mode,heartbeat,out/'events.jsonl')
    write_quotes(out/'normalized.csv',q);save_json(out/'diffusion.json',config)
    if refinements<1:raise ValueError('At least one proof refinement required')
    if storage_stride<1:raise ValueError('Positive temporal storage stride required')
    proof=None
    for level in range(refinements):
        grid=plateau.make_grid(config,spatial_cells*2**level,pulse_cells*2**level)
        if len(grid)>20000:raise ValueError('Fixed-model mesh exceeds the explicit node budget')
        arrays,times=plateau.solve_numerical(config,grid,time_steps*2**level,progress)
        if storage_stride>1:
            compact={};retained=[]
            for p,ts in enumerate(times):
                ids=sorted(set(range(min(12,len(ts))))|set(range(12,len(ts),storage_stride))|{len(ts)-1})
                compact[f'phase_{p}']=arrays[f'phase_{p}'][ids].copy();retained.append([ts[i] for i in ids])
            arrays,times=compact,retained
            progress.event('fixed_model_payload','compacted',stride=storage_stride,
                note='All residuals are reverified on the retained larger time cells')
        np.savez_compressed(out/'surface.tmp.npz',**arrays);(out/'surface.tmp.npz').replace(out/'surface.npz')
        save_json(out/'mesh.json',dict(grid=list(map(str,grid)),times=times))
        proof=plateau.certify_prices(config,grid,times,arrays,q,progress);save_json(out/'pricing.json',proof)
        progress.event('fixed_model_price_proof','done',done=level+1,total=refinements,compatible=proof['all_quotes_compatible'])
        if proof['all_quotes_compatible']:break
    save_json(out/'model_manifest.json',dict(version=__version__,status='completed',quotes=quote_payload(q),
        compatible=proof['all_quotes_compatible'],artifacts={name:sha256(out/name) for name in FILES}))
    if proof['all_quotes_compatible']:
        _VERIFIED_MODELS[proof_key(out)]=dict(verified=True,compatible=True,pde_solver_restarted=False,optimizer_restarted=False)
    return proof


def verify(directory,progress=QUIET):
    out=Path(directory)
    if (out/'exact_manifest.json').exists():
        result=exact_endpoint.verify(out,progress)
        proof=json.loads((out/'pricing.json').read_text(encoding='utf-8'))
        if not proof['all_quotes_compatible']:raise ValueError('Exact model is not price compatible')
        return result
    m=json.loads((out/'model_manifest.json').read_text(encoding='utf-8'))
    if m.get('proof_kind')=='analytic_piecewise_time_clock_v1':
        from . import clock_models
        return clock_models.verify(out)
    if m['status']!='completed' or set(m['artifacts'])!=set(FILES):raise ValueError('Incomplete fixed diffusion certificate')
    for name,digest in m['artifacts'].items():
        if sha256(out/name)!=digest:raise ValueError('Changed fixed model: '+name)
    q=read_quotes(out/'normalized.csv');config=json.loads((out/'diffusion.json').read_text(encoding='utf-8'))
    if quote_payload(q)!=m['quotes']:raise ValueError('Fixed model uses different observations')
    mesh=json.loads((out/'mesh.json').read_text(encoding='utf-8'));grid=list(map(F,mesh['grid']))
    with np.load(out/'surface.npz',allow_pickle=False) as arrays:
        proof=plateau.certify_prices(config,grid,mesh['times'],arrays,q,progress)
    if proof!=json.loads((out/'pricing.json').read_text(encoding='utf-8')) or proof['all_quotes_compatible']!=m['compatible']:
        raise ValueError('Incorrect fixed-model pricing claim')
    if not m['compatible']:raise ValueError('Model bank requires certified compatible diffusions')
    return dict(verified=True,compatible=True,pde_solver_restarted=False,optimizer_restarted=False)


def variance_at(config,phase,x):
    knots=plateau.phase_knots(config,phase);values=list(map(F,config['phases'][phase]['variance']))
    if x<=knots[0]:return values[0]
    if x>=knots[-1]:return values[-1]
    j=next(i for i in range(len(knots)-1) if knots[i]<=x<=knots[i+1])
    u=(x-knots[j])/(knots[j+1]-knots[j]);s=10*u**3-15*u**4+6*u**5
    return values[j]+(values[j+1]-values[j])*s


def coefficient_range(config,family):
    ta,tb,ka,kb=family.support();values=[]
    t0=F(config['prefix_end']);last=F(config['phases'][-1]['right'])
    if ta<t0 or tb>last:values.append(F(config['prefix_variance']))
    for p,phase in enumerate(config['phases']):
        if max(ta,F(phase['left']))>=min(tb,F(phase['right'])):continue
        ks=[ka,kb]+[k for k in plateau.phase_knots(config,p) if ka<k<kb]
        values.extend(variance_at(config,p,k) for k in ks)
    if not values:raise ValueError('Empty positive kernel support')
    return I(min(values),max(values))


@lru_cache(maxsize=32768)
def moments(poly,a,b,left,right):
    lo,hi=max(a,left),min(b,right)
    if lo>=hi:return (F(0),)*4
    answer=[]
    for j in range(4):
        basis=[F(comb(3,j),1)/(b-a)**3]
        for unused in range(j):basis=mul(basis,[-a,1])
        for unused in range(3-j):basis=mul(basis,[b,-1])
        answer.append(integrate(mul(poly,basis),lo,hi))
    return tuple(answer)


def abs_integral(poly,a,b,depth=10):
    r=bernstein_range(poly,a,b)
    if r.lo>=0:return integrate(poly,a,b)
    if r.hi<=0:return -integrate(poly,a,b)
    if depth==0:return (b-a)*max(abs(r.lo),abs(r.hi))
    m=(a+b)/2
    return abs_integral(poly,a,m,depth-1)+abs_integral(poly,m,b,depth-1)


def bezier(values,grid,times):
    dx=bi.stack([B.rational(b-a) for a,b in zip(grid,grid[1:])],0)
    h=bi.stack([B.rational(b-a) for a,b in zip(times,times[1:])],0)
    def space(v,s):
        v,s=B(v),B(s)
        return bi.stack([v[...,:-1],v[...,:-1]+dx*s[...,:-1]/3,v[...,1:]-dx*s[...,1:]/3,v[...,1:]],-1)
    left=space(values[:-1,0],values[:-1,1]);right=space(values[1:,0],values[1:,1])
    lt=space(values[:-1,2],values[:-1,3]);rt=space(values[1:,2],values[1:,3])
    return bi.stack([left,left+h[:,None,None]*lt/3,right-h[:,None,None]*rt/3,right],-2)


class Model:
    """An already verified model, loaded once per bank verification."""
    def __init__(self,folder):
        self.folder=Path(folder);self.config=json.loads((self.folder/'diffusion.json').read_text(encoding='utf-8'))
        self.pricing=json.loads((self.folder/'pricing.json').read_text(encoding='utf-8'))
        self.quotes=read_quotes(self.folder/'normalized.csv')
        self.analytic_clock=self.pricing.get('kind')=='analytic_piecewise_time_clock_v1'
        self.cache={}
        if self.analytic_clock:
            self.grid=[];self.times=[];self.arrays={}
            return
        m=json.loads((self.folder/'mesh.json').read_text(encoding='utf-8'));self.grid=list(map(F,m['grid']));self.times=[list(map(F,t)) for t in m['times']]
        with np.load(self.folder/'surface.npz',allow_pickle=False) as z:self.arrays={k:z[k].copy() for k in z.files}
        self.cache={}

    def price_error(self,time):
        if self.analytic_clock:return F(0)
        c=self.config;t0=F(c['prefix_end']);a0=F(c['prefix_variance']);amax=F(c['amax'])
        v=plateau.variance_envelope(c,time)[1]
        boundary=max(black_call(self.grid[0],v).hi-(1-self.grid[0]),black_call(self.grid[-1],v).hi)
        offset=abs(F(float(self.arrays['phase_0'][0,0,0]))-(1-self.grid[0]))
        boundary+=offset
        residual=sum(F(e) for e,p in zip(self.pricing['residual_integrals'],c['phases']) if F(p['left'])<time)
        return max(F(self.pricing['initial_error']),boundary)+residual

    def point_average(self,w,force_weak=False):
        key=(tuple(w.json().values()),bool(force_weak))
        if key in self.cache:return self.cache[key]
        from .families import Family
        c=self.config;amin,amax=F(c['amin']),F(c['amax']);cr=coefficient_range(c,Family.point(w))
        if cr.lo==cr.hi and not force_weak:
            answer=dict(method='exact_constant_coefficient_on_support',average=cr.json(),denominator=None)
            self.cache[key]=answer;return answer
        if self.analytic_clock:
            answer=dict(method='analytic_clock_coefficient_range',average=cr.json(),denominator=None)
            self.cache[key]=answer;return answer
        ta,tb,ka,kb=w.t-w.ht,w.t+w.ht,w.k-w.hk,w.k+w.hk
        if ta<F(c['prefix_end']) or tb>F(c['phases'][-1]['right']) or ka<self.grid[0] or kb>self.grid[-1]:
            answer=dict(method='coefficient_range_only',average=cr.json(),denominator=None)
            self.cache[key]=answer;return answer
        pt,pk=bump(w.t,w.ht),bump(w.k,w.hk);nt=tuple(-2*x for x in deriv(pt));dk=tuple(deriv(deriv(mul([0,0,1],pk))))
        ik=[j for j,(a,b) in enumerate(zip(self.grid,self.grid[1:])) if max(a,ka)<min(b,kb)]
        kgrid=self.grid[ik[0]:ik[-1]+2]
        nk=bi.stack([bi.stack([B.rational(v) for v in moments(tuple(pk),a,b,ka,kb)],-1) for a,b in zip(kgrid,kgrid[1:])],0)
        dkw=bi.stack([bi.stack([B.rational(v) for v in moments(dk,a,b,ka,kb)],-1) for a,b in zip(kgrid,kgrid[1:])],0)
        numerator=B(0);denom=B(0)
        for p,times in enumerate(self.times):
            it=[j for j,(a,b) in enumerate(zip(times,times[1:])) if max(a,ta)<min(b,tb)]
            if not it:continue
            selected=times[it[0]:it[-1]+2]
            z=self.arrays[f'phase_{p}'][it[0]:it[-1]+2,:,ik[0]:ik[-1]+2]
            v=bezier(z,kgrid,selected)
            tn=bi.stack([bi.stack([B.rational(x) for x in moments(nt,a,b,ta,tb)],-1) for a,b in zip(selected,selected[1:])],0)
            td=bi.stack([bi.stack([B.rational(x) for x in moments(tuple(pt),a,b,ta,tb)],-1) for a,b in zip(selected,selected[1:])],0)
            numerator=numerator+bi.sum_all(v*tn[:,None,:,None]*nk[None,:,None,:])
            denom=denom+bi.sum_all(v*td[:,None,:,None]*dkw[None,:,None,:])
        epsilon=self.price_error(tb)
        nerror=F(128,35)*w.hk*epsilon
        derror=F(32,35)*w.ht*abs_integral(list(dk),ka,kb)*epsilon
        n=I(F(float(numerator.lo))-nerror,F(float(numerator.hi))+nerror)
        d=I(F(float(denom.lo))-derror,F(float(denom.hi))+derror)
        if d.lo>0:
            a=(n/d).intersect(cr)
            answer=dict(method='weak_price_functional_with_PDE_error',average=a.json(),denominator=d.json(),
                numerator=n.json(),price_error=str(epsilon),numerator_error=str(nerror),denominator_error=str(derror))
        else:
            answer=dict(method='coefficient_range_only_weak_denominator_unresolved',average=cr.json(),denominator=d.json(),
                numerator=n.json(),price_error=str(epsilon),numerator_error=str(nerror),denominator_error=str(derror))
        self.cache[key]=answer;return answer

    def family_average(self,family):
        cr=coefficient_range(self.config,family)
        if cr.lo==cr.hi:return dict(method='exact_coefficient_range',average=cr.json(),denominator=None)
        point=self.point_average(family.center());a=I(F(point['average']['lo']),F(point['average']['hi']))
        if point['denominator'] is None:return dict(method='coefficient_range_only',average=cr.json(),point=point)
        dmin=F(point['denominator']['lo']);ta,tb,ka,kb=family.support()
        deviation=family.weight_deviation();energy=F(self.config['amax'])*tb
        if deviation and energy>100:
            return dict(method='coefficient_range_only_moment_arithmetic_budget',average=cr.json(),point=point)
        error=deviation*(tb-ta)*exp_point(energy).hi if deviation else F(0)
        lower=dmin-error
        if lower<=0:return dict(method='coefficient_range_only_family_denominator_unresolved',average=cr.json(),point=point,weight_mass_error=str(error))
        radius=(F(self.config['amax'])-F(self.config['amin']))*error/lower
        answer=(a+I(-radius,radius)).intersect(cr)
        return dict(method='uniform_weight_continuity',average=answer.json(),point=point,
                    weight_mass_error=str(error),family_denominator_lower=str(lower),average_radius=str(radius))
