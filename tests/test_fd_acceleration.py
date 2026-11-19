"""Check the actual CUDA algorithm against independent LAPACK SDIRK2.

The host emulator exercises the same source and barriers with OpenMP. It is
not a substitute for the optional real GPU execution/parity test.
"""
import ctypes as C
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from dupire_identify.fd_acceleration import cuda_source, CudaFD, BatchEngine
from dupire_identify.calibrate_general import fd_batch_python, black


@pytest.fixture(scope='module')
def emulated_kernel(tmp_path_factory):
    compiler=shutil.which('g++')
    if not compiler or os.name=='nt':pytest.skip('Optional g++/OpenMP kernel host emulator')
    folder=tmp_path_factory.mktemp('pcr_host')
    prefix='''#include <cmath>
#include <vector>
#include <omp.h>
struct Dim {int x;};
static thread_local Dim threadIdx,blockIdx,blockDim;
static double* sim_shared;
#define __global__
#define __syncthreads() _Pragma("omp barrier")
'''
    source=cuda_source().replace('extern __shared__ double sh[];','double* sh=sim_shared;')
    suffix='''
extern "C" void simulate(int Bn,int P,int N,int levels,int S,const double* grid,
const double* coeff,const double* lefts,const double* rights,const int* counts,
const double* initial,const double* window,double* factors,double* curves,double* nums,double* dens){
std::vector<double> shared(4*S+4);sim_shared=shared.data();omp_set_dynamic(0);
for(int model=0;model<Bn;++model){
#pragma omp parallel num_threads(S)
{threadIdx.x=omp_get_thread_num();blockIdx.x=model;blockDim.x=S;
fd_batch_kernel(Bn,P,N,levels,grid,coeff,lefts,rights,counts,initial,window,factors,curves,nums,dens);}
}}
'''
    (folder/'kernel.cpp').write_text(prefix+source+suffix)
    subprocess.run([compiler,'-shared','-fPIC','-fopenmp','-O2','-ffp-contract=off',
        str(folder/'kernel.cpp'),'-o',str(folder/'kernel.so')],check=True,capture_output=True)
    library=C.CDLL(str(folder/'kernel.so'));func=library.simulate
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    iptr=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    func.argtypes=[C.c_int]*5+[ptr]*4+[iptr]+[ptr]*6
    def invoke(grid,coeff,lefts,rights,initial,steps,window):
        Bn,P,N=coeff.shape;M=N-2;S=1<<(M-1).bit_length();levels=(S-1).bit_length()
        counts=np.maximum(1,np.ceil((rights-lefts)*steps)).astype(np.int32)
        factors=np.empty((Bn,levels,2,M));curves=np.empty_like(coeff);nums=np.empty(Bn);dens=np.empty(Bn)
        func(Bn,P,N,levels,S,grid,coeff,lefts,rights,counts,initial,window,factors,curves,nums,dens)
        return curves,nums,dens
    return invoke


@pytest.mark.parametrize('nodes',[9,17,19])
def test_actual_pcr_source_matches_lapack_on_nonuniform_mesh(emulated_kernel,nodes):
    rng=np.random.default_rng(865)
    x=np.geomspace(.3,3.,nodes);lefts=np.array([.125,.28]);rights=np.array([.28,.6])
    coeff=np.ascontiguousarray(rng.uniform(.005,.22,size=(2,2,nodes)))
    initial=black(x[None,:],np.array([.04,.08])[:,None]*lefts[0]);initial[:,0]=1-x[0];initial[:,-1]=0.
    initial=np.ascontiguousarray(initial);window=np.array([.35,1.,.18,.5])
    args=(x,coeff,lefts,rights,initial,16,window)
    actual=emulated_kernel(*args);expected=fd_batch_python(*args)
    for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-13)


def test_real_cuda_batch_parity():
    cp=pytest.importorskip('cupy')
    try:devices=cp.cuda.runtime.getDeviceCount()
    except Exception as exc:pytest.skip('No CUDA driver: '+str(exc))
    if not devices:pytest.skip('No CUDA device')
    report=CudaFD().diagnostics
    assert report['max_abs_curve_difference']<2e-11


def test_auto_rejects_slower_device_even_when_numerically_valid(monkeypatch):
    import dupire_identify.fd_acceleration as m
    class Fake:
        def __init__(self,progress):pass
        def __call__(self,*args):return fd_batch_python(*args)
    monkeypatch.setattr(m,'CudaFD',Fake)
    engine=BatchEngine('auto');engine.native=None
    ticks=iter([0.,1.,2.,5.])
    monkeypatch.setattr(m.time,'perf_counter',lambda:next(ticks))
    x=np.geomspace(.4,2.,9);c=np.full((2,1,9),.04);ls=np.array([.1]);rs=np.array([.2]);init=black(x[None,:],np.array([.004,.004])[:,None])
    init[:,0]=1-x[0];init[:,-1]=0.
    engine(x,c,ls,rs,init,10,np.array([.15,1.,.05,.2]))
    assert engine.selected=='numpy' and engine.timing['cuda_s']>engine.timing['cpu_s']
