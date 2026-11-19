"""Guarded binary64 acceleration of numerical diffusion proposals.

CUDA is never used for outward interval arithmetic or rational proofs. Every
device is tested against independent LAPACK SDIRK2 before use. Auto mode also
times a representative batch and retains CPU when CUDA is slower.
"""
from pathlib import Path
import time
import numpy as np
from .progress import QUIET


def cuda_source():
    return (Path(__file__).parent/'kernels/fd_batch.cu').read_text(encoding='utf-8')


class CudaFD:
    def __init__(self, progress=QUIET):
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('No CUDA device for batched diffusion proposals')
        self.cp=cp;self.progress=progress;self.geometry={}
        with progress.phase('cuda_fd_compilation'):
            self.kernel=cp.RawKernel(cuda_source(),'fd_batch_kernel',
                options=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true'))
            self.kernel.compile()
        self.diagnostics=self.self_test()
        progress.event('cuda_fd_parity','verified',**self.diagnostics)

    def __call__(self,grid,coefficients,lefts,rights,initial,steps,window):
        cp=self.cp
        coeff=np.ascontiguousarray(coefficients,dtype=np.float64)
        grid=np.ascontiguousarray(grid,dtype=np.float64)
        initial=np.ascontiguousarray(initial,dtype=np.float64)
        Bn,P,N=coeff.shape;M=N-2;block=1<<(M-1).bit_length();levels=(block-1).bit_length()
        if (M<1 or block>1024 or initial.shape!=(Bn,N) or len(grid)!=N or
            len(lefts)!=P or len(rights)!=P or steps<=0 or np.any(np.diff(grid)<=0)):
            raise ValueError('Unsupported CUDA batch geometry (at most 1024 interior nodes)')
        counts=np.maximum(1,np.ceil((np.asarray(rights)-np.asarray(lefts))*steps)).astype(np.int32)
        if np.any(~np.isfinite(coeff)) or np.any(~np.isfinite(initial)):
            raise ValueError('Nonfinite CUDA proposal input')
        key=(grid.tobytes(),np.asarray(lefts).tobytes(),np.asarray(rights).tobytes(),counts.tobytes(),np.asarray(window).tobytes())
        if key not in self.geometry:
            self.geometry={key:tuple(cp.asarray(a) for a in (grid,np.asarray(lefts,dtype=np.float64),
                np.asarray(rights,dtype=np.float64),counts,np.asarray(window,dtype=np.float64)))}
        x,ls,rs,cs,w=self.geometry[key]
        cg=cp.asarray(coeff);init=cp.asarray(initial)
        factors=cp.empty((Bn,levels,2,M),dtype=cp.float64)
        curves=cp.empty((Bn,P,N),dtype=cp.float64);nums=cp.empty(Bn,dtype=cp.float64);dens=cp.empty_like(nums)
        self.kernel((Bn,),(block,),(np.int32(Bn),np.int32(P),np.int32(N),np.int32(levels),
            x,cg,ls,rs,cs,init,w,factors,curves,nums,dens),shared_mem=(4*block+4)*8)
        result=tuple(a.get() for a in (curves,nums,dens))
        if any(np.any(~np.isfinite(a)) for a in result):
            raise RuntimeError('Nonfinite CUDA FD output; no model may be certified from this proposal')
        return result

    def self_test(self):
        from .calibrate_general import fd_batch_python,black
        x=np.geomspace(.25,4.,39);ls=np.array([.125,.3]);rs=np.array([.3,.6])
        c=np.array([np.tile(.03+.01/(1+x),(2,1)),np.tile(.04+.02*x/(1+x),(2,1))])
        c[:,1]*=1.3
        init=black(x[None,:],np.array([.04,.06])[:,None]*ls[0]);init[:,0]=1-x[0];init[:,-1]=0.
        w=np.array([.35,1.,.16,.3]);args=(x,c,ls,rs,init,40,w)
        ref=fd_batch_python(*args);actual=self(*args)
        errors=[float(np.max(abs(a-b))) for a,b in zip(actual,ref)]
        if any(e>2e-11 for e in errors):
            raise RuntimeError('CUDA FD parity failed: '+str(errors))
        return dict(max_abs_curve_difference=errors[0],max_abs_numerator_difference=errors[1],
                    max_abs_denominator_difference=errors[2],dtype='float64',fast_math=False,
                    certificates_on_cpu=True)


class BatchEngine:
    def __init__(self,requested='auto',threads=1,progress=QUIET):
        if requested not in ('auto','numpy','cpp','cuda'):
            raise ValueError('Unknown batch backend')
        self.requested=requested;self.threads=threads;self.progress=progress;self.cuda=None
        self.selected=None;self.timing=None
        self.choices={}
        from .calibrate_general import _native_batch
        self.native=_native_batch if requested!='numpy' else None
        if requested=='cpp' and self.native is None:
            raise RuntimeError('C++ FD extension missing; build_install.py --native')
        if requested in ('auto','cuda'):
            try:self.cuda=CudaFD(progress)
            except Exception as exc:
                progress.event('cuda_fd','unavailable',reason=str(exc))
                if requested=='cuda':raise
        if self.cuda is None:self.selected='cpp' if self.native else 'numpy'
        elif requested=='cuda':self.selected='cuda'

    def cpu(self,*args):
        from .calibrate_general import fd_batch_python
        return self.native(*args,self.threads) if self.native else fd_batch_python(*args)

    def __call__(self,*args):
        if self.cuda is not None and args[1].shape[2]-2>1024:
            self.progress.event('fd_batch','cpu_geometry_fallback',nodes=args[1].shape[2])
            return self.cpu(*args)
        bucket=(args[1].shape[0]-1).bit_length()
        if self.cuda is not None and self.requested=='auto':self.selected=self.choices.get(bucket)
        if self.selected is None:
            # Include transfers; exclude JIT (already compiled and exercised).
            start=time.perf_counter();cpu=self.cpu(*args);cpu_s=time.perf_counter()-start
            start=time.perf_counter();gpu=self.cuda(*args);cuda_s=time.perf_counter()-start
            errors=[float(np.max(abs(a-b))) for a,b in zip(gpu,cpu)]
            if any(not np.isfinite(e) or e>2e-10 for e in errors):
                self.cuda=None;self.selected='cpp' if self.native else 'numpy'
                self.progress.event('fd_batch','cpu_parity_fallback',max_abs_differences=errors)
                return cpu
            self.selected='cuda' if cuda_s<cpu_s else 'cpp' if self.native else 'numpy'
            self.choices[bucket]=self.selected
            self.timing=dict(cpu_s=cpu_s,cuda_s=cuda_s,max_abs_differences=errors,transfers_included=True)
            self.progress.event('fd_batch_autotune','selected',backend=self.selected,**self.timing)
            return gpu if self.selected=='cuda' else cpu
        return self.cuda(*args) if self.selected=='cuda' else self.cpu(*args)


def benchmark(backend='auto',models=64,nodes=320,repeats=3,threads=1,progress=QUIET):
    from .calibrate_general import black
    if min(models,repeats)<1 or nodes<5:raise ValueError('Positive batch/repeats and at least five nodes required')
    x=np.geomspace(.125,8.,nodes);lefts=np.array([.125,.25,.5]);rights=np.array([.25,.5,1.])
    variance=np.linspace(.02,.09,models)
    coefficients=np.ascontiguousarray(variance[:,None,None]*(1+.15*np.arange(3)[None,:,None])*(1+.1*x[None,None,:]/(1+x[None,None,:])))
    initial=black(x[None,:],variance[:,None]*lefts[0]);initial[:,0]=1-x[0];initial[:,-1]=0.
    args=(x,coefficients,lefts,rights,np.ascontiguousarray(initial),256,np.array([.4,1.,.12,.2]))
    engine=BatchEngine(backend,threads,progress)
    reference=engine.cpu(*args);engine(*args)
    cpu_times=[];times=[];difference=[]
    for i in range(repeats):
        start=time.perf_counter();reference=engine.cpu(*args);cpu_times.append(time.perf_counter()-start)
        start=time.perf_counter();actual=engine(*args);times.append(time.perf_counter()-start)
        difference=[float(np.max(abs(a-b))) for a,b in zip(actual,reference)]
        if any(e>2e-10 for e in difference):raise RuntimeError('FD benchmark numerical parity failed: '+str(difference))
        progress.event('fd_benchmark','running',done=i+1,total=repeats,backend=engine.selected)
    cpu=float(np.median(cpu_times));chosen=float(np.median(times))
    return dict(backend=engine.selected,models=models,nodes=nodes,repeats=repeats,
        cpu_median_s=cpu,selected_median_s=chosen,measured_speedup=cpu/chosen,
        cpu_timings_s=cpu_times,selected_timings_s=times,max_abs_differences=difference,
        includes_host_device_transfers=True,warmup_and_compilation_excluded=True,
        precision='float64, no fast math',workload='batched SDIRK2 diffusion proposals, not rational certificates')
