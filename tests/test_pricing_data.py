from fractions import Fraction as F
import csv
import json
import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path
import runpy
import numpy as np
import pytest
from dupire_identify.backend import Backend
from dupire_identify.io import read_quotes
from dupire_identify.data import normalize,estimate_curves
from dupire_identify.p3 import common_american_models


def price(engine,american=False,dividend=0.,nx=161,steps=160):
    times=np.unique(np.r_[np.linspace(0,1,steps+1),.5])
    div=np.zeros(len(times));div[np.searchsorted(times,.5)]=dividend
    return engine.price([[.04,0.],[.06,-.25]],[.9,1.,1.1],[1,0,1],
                        [len(times)-1]*3,times,div,1.,.03,nx=nx,american=american,batch=1)


def test_numerical_pde_against_black_and_exercise():
    from scipy.special import ndtr
    engine=Backend('numpy')
    european=price(engine)
    d1=(np.log(1/.9)+(.03+.02))/.2;d2=d1-.2
    expected=ndtr(d1)-.9*np.exp(-.03)*ndtr(d2)
    assert abs(european[0,0]-expected)<.0025
    american=price(engine,True)
    assert np.all(american>=european-1e-12)
    assert abs(american[0,0]-european[0,0])<1e-10
    assert american[0,1]>european[0,1]
    dividend=price(engine,True,.1)
    assert np.all(np.isfinite(dividend))
    assert dividend[0,0]>0


@pytest.mark.skipif(importlib.util.find_spec('dupire_identify._native') is None,reason='optional C++ not built')
def test_cpp_matches_numpy_with_dividends():
    engine=Backend('cpp',threads=2)
    a=price(engine,True,.08,nx=81,steps=60)
    b=price(Backend('numpy'),True,.08,nx=81,steps=60)
    assert np.max(np.abs(a-b))<5e-12
    ks=np.geomspace(.4,2.5,71);vs=np.linspace(.001,.7,31)
    assert np.max(np.abs(engine.black(ks,vs)-Backend('numpy').black(ks,vs)))<5e-14


def test_explicit_cuda_does_not_silently_fall_back(monkeypatch):
    import builtins
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name=='cupy':raise ImportError('test no GPU')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(RuntimeError,match='explicitly requested'):
        Backend('cuda')


@pytest.mark.parametrize('fail',[False,True])
def test_cuda_is_ready_only_after_production_compilation(monkeypatch,fail):
    class Array(np.ndarray):
        def get(self):return np.asarray(self)
    names=[]
    class Module:
        def get_function(self,name):
            names.append(name)
            if fail:raise RuntimeError('NVRTC cannot open source file math.h')
            return object()
    def probe(*args):
        def launch(grid,block,parameters):parameters[0][0]+=1
        return launch
    fake=SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda:1),
                             Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda:None))),
        RawKernel=probe,RawModule=lambda **kwargs:Module(),float64=np.float64,
        zeros=lambda *a,**kw:np.zeros(*a,**kw).view(Array))
    monkeypatch.setitem(sys.modules,'cupy',fake)
    if fail:
        with pytest.raises(RuntimeError,match='math.h'):Backend('cuda')
        fallback=Backend('auto')
        assert fallback.name in ('cpp','numpy') and fallback.cp is None
        assert any('math.h' in x for x in fallback.diagnostics)
    else:
        engine=Backend('cuda')
        assert engine.name=='cuda'
        assert set(names)=={'price_kernel','black_kernel'}
        original=engine.module
        assert engine._cuda_module() is original


def test_repair_script_backs_up_and_is_idempotent(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts'/'repair_cuda_nvrtc.py'
    repair=runpy.run_path(str(script))['repair']
    header=tmp_path/'pricer.h'
    original=b'#include <math.h>\ninline double f(double x) { return sqrt(x); }\n'
    header.write_bytes(original)
    assert repair(header)
    updated=header.read_bytes()
    assert b'#ifndef __CUDACC_RTC__' in updated
    assert header.with_name('pricer.h.before-nvrtc-fix').read_bytes()==original
    assert not repair(header)
    assert header.read_bytes()==updated


def test_cuda_matches_cpu_when_available():
    try:engine=Backend('cuda')
    except RuntimeError:pytest.skip('CUDA device/runtime unavailable here')
    a=price(engine,True,.08,nx=81,steps=60)
    b=price(Backend('numpy'),True,.08,nx=81,steps=60)
    assert np.max(np.abs(a-b))<5e-11
    assert np.max(np.abs(engine.black([.8,1,1.2],[.01,.04,.09])-
                         Backend('numpy').black([.8,1,1.2],[.01,.04,.09])))<5e-13


def test_input_validation_before_native_kernel():
    with pytest.raises(ValueError):
        Backend('numpy').price([[.04,0]],[1],[1],[1],[0,1],[0,0],1.,nx=2)
    with pytest.raises(ValueError):Backend('numpy').black([0],[.04])


def test_normalization_is_exact_and_rejects_mixed_conventions(tmp_path):
    raw=tmp_path/'raw.csv';curves=tmp_path/'curves.csv';out=tmp_path/'q.csv'
    row=dict(contract='x',style='european',kind='call',t='1/2',strike='100',bid='5',ask='6',
             expiry='2027-01-15',root='SPXW',settlement='PM',symbol='^SPX',snapshot_utc='2026-11-09T15:00:00Z')
    with raw.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
    curves.write_text('expiry,root,settlement,forward,discount\n2027-01-15,SPXW,PM,101,0.99\n')
    assert normalize(raw,curves,out)==1
    q=read_quotes(out)[0]
    assert q.k==F(100,101) and q.bid==F(500,9999)
    row['style']='american'
    with raw.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
    with pytest.raises(ValueError,match='American'):
        normalize(raw,curves,out)


def test_p3_retains_common_vectors_and_labels_numerical_scope(tmp_path):
    quotes=tmp_path/'a.csv';divs=tmp_path/'d.json'
    quotes.write_text('id,t,strike,bid,ask,kind,style\na,0.5,0.9,0,1,call,american\nb,1,1.1,0,1,put,american\n')
    divs.write_text('[{"t":0.4,"amount":0.02}]')
    r=common_american_models(quotes,divs,tmp_path/'p3',1.,variances=[.04,.06],skews=[0],
                             nx=41,steps=20,backend='numpy',refine=True)
    assert r['accepted_models']==2
    assert r['common_model_preserved']
    assert not r['continuous_american_certified']
    assert not r['full_inverse_set_covered']
    with np.load(tmp_path/'p3'/'common_models.npz') as z:
        assert z['european'].shape==(2,2)
        assert np.all(z['american']>=z['european']-1e-12)
