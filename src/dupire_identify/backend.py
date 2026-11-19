"""Accelerated exploratory work; certificates are verified on CPU rationals."""
from pathlib import Path
import os
import numpy as np
from scipy.special import ndtr
from scipy.linalg import solve_banded
from .progress import QUIET


def cuda_source():
    """The same self-contained source for CuPy and driver-free NVRTC tests."""
    root=Path(__file__).parent/"kernels"
    return (root/"price.cu").read_text(encoding="utf-8").replace(
        '#include "pricer.h"',(root/"pricer.h").read_text(encoding="utf-8"))


class Backend:
    def __init__(self, requested="auto", threads=None, progress=QUIET):
        if requested not in ("auto","numpy","cpp","cuda"):
            raise ValueError("backend must be auto/numpy/cpp/cuda")
        self.threads=max(1,(os.cpu_count() or 2)-2) if threads is None else threads
        if self.threads<1: raise ValueError("threads must be positive")
        self.progress=progress
        self.native=None;self.cp=None;self.module=None;self.name="numpy"
        self.diagnostics=[]
        if requested in ("auto","cuda"):
            try:
                import cupy as cp
                if cp.cuda.runtime.getDeviceCount()<1: raise RuntimeError("No CUDA device")
                # Test actual compilation and execution, not just device presence.
                probe=cp.RawKernel('extern "C" __global__ void p(double*x){x[0]+=1;}',"p")
                a=cp.zeros(1,dtype=cp.float64);probe((1,),(1,),(a,));cp.cuda.Stream.null.synchronize()
                if float(a.get()[0])!=1: raise RuntimeError("CUDA probe failed")
                self.cp=cp
                # Compile both production kernels before declaring CUDA ready.
                # A trivial probe alone misses platform-specific source errors.
                self._cuda_module()
                self.name="cuda"
            except Exception as e:
                self.cp=None;self.module=None
                self.diagnostics.append("cuda: "+str(e))
                if requested=="cuda": raise RuntimeError("CUDA explicitly requested but unavailable: "+str(e)) from e
        if self.cp is None and requested in ("auto","cpp"):
            try:
                from . import _native
                self.native=_native;self.name="cpp"
            except ImportError as e:
                self.diagnostics.append("cpp: optional native extension not built")
                if requested=="cpp": raise RuntimeError("C++ extension missing; set DUPIRE_NATIVE=1 before pip install .") from e
        self.progress.event("backend","selected",backend=self.name,threads=self.threads,
                            diagnostics=self.diagnostics)

    def _cuda_module(self):
        if self.module is None:
            with self.progress.phase("cuda_kernel_compilation"):
                candidate=self.cp.RawModule(code=cuda_source(),options=("--std=c++11",),backend="nvrtc")
                candidate.get_function("price_kernel")
                candidate.get_function("black_kernel")
                self.module=candidate
        return self.module

    def black(self,strikes,variances):
        k=np.ascontiguousarray(strikes,dtype=np.float64)
        v=np.ascontiguousarray(variances,dtype=np.float64)
        if k.ndim!=1 or v.ndim!=1 or not len(k) or not len(v) or np.any(k<=0) or np.any(v<0):
            raise ValueError("Positive 1D strikes and nonnegative 1D variances required")
        if not np.all(np.isfinite(k)) or not np.all(np.isfinite(v)): raise ValueError("Nonfinite Black inputs")
        if self.native: return self.native.black(k,v,self.threads)
        if self.cp is not None:
            cp=self.cp;kg=cp.asarray(k);vg=cp.asarray(v);out=cp.empty((len(v),len(k)),dtype=cp.float64)
            n=out.size
            self._cuda_module().get_function("black_kernel")(((n+255)//256,),(256,),
                (np.int64(n),np.int32(len(k)),kg,vg,out))
            return out.get()
        s=np.sqrt(np.maximum(v[:,None],1e-300))
        d1=(-np.log(k[None,:])+v[:,None]/2)/s
        out=ndtr(d1)-k[None,:]*ndtr(d1-s)
        out[v==0]=np.maximum(1-k,0)
        return out

    def price(self,models,strikes,iscall,maturity_index,times,dividends,
              spot,rate=0.,yield_=0.,smax=None,nx=257,american=True,batch=64):
        models=np.ascontiguousarray(models,dtype=np.float64)
        strikes=np.ascontiguousarray(strikes,dtype=np.float64)
        call=np.ascontiguousarray(iscall,dtype=np.int32)
        mi=np.ascontiguousarray(maturity_index,dtype=np.int32)
        times=np.ascontiguousarray(times,dtype=np.float64)
        div=np.ascontiguousarray(dividends,dtype=np.float64)
        spot=float(spot);smax=float(smax or 4*max(spot,float(strikes.max())))
        if models.ndim!=2 or models.shape[1]!=2 or len(models)==0 or len(strikes)==0:
            raise ValueError("Expected models[B,2] and nonempty contracts")
        if not all(np.all(np.isfinite(x)) for x in (models,strikes,times,div)):
            raise ValueError("Nonfinite pricing input")
        if (np.any(models[:,0]<=0) or np.any(np.abs(models[:,1])>=1) or spot<=0 or
            nx<5 or smax<=spot or np.any(strikes<=0) or batch<1 or rate<0 or yield_<0 or
            not all(np.isfinite(x) for x in (spot,rate,yield_,smax)) or
            times[0]!=0 or np.any(np.diff(times)<=0) or len(div)!=len(times) or
            np.any(div<0) or div[0]!=0 or len(call)!=len(strikes) or len(mi)!=len(strikes) or
            np.any(mi<=0) or np.any(mi>=len(times)) or np.any((call!=0)&(call!=1))):
            raise ValueError("Invalid model/domain/time/dividend/contract input (r,q>=0 required)")
        outputs=[]
        for start in range(0,len(models),batch):
            mod=models[start:start+batch]
            if self.native:
                out=self.native.price(mod,strikes,call,mi,times,div,spot,float(rate),float(yield_),
                                      smax,nx,bool(american),self.threads)
            elif self.cp is not None:
                cp=self.cp;n=len(mod)*len(strikes)
                # Leave substantial VRAM headroom, shrink user batch if needed.
                memory=n*nx*6*8
                free,_=cp.cuda.runtime.memGetInfo()
                if memory>free*.65:
                    raise MemoryError(f"Batch needs {memory/2**20:.1f} MiB; lower --batch")
                arrays=[cp.asarray(x) for x in (mod,strikes,call,mi,times,div)]
                work=cp.empty(n*nx*6,dtype=cp.float64);o=cp.empty(n,dtype=cp.float64)
                args=(np.int32(n),arrays[0],np.int32(len(strikes)),*arrays[1:],
                      np.float64(spot),np.float64(rate),np.float64(yield_),np.float64(smax),
                      np.int32(nx),np.int32(american),work,o)
                self._cuda_module().get_function("price_kernel")(((n+127)//128,),(128,),args)
                out=o.get().reshape(len(mod),len(strikes))
                del work,o,arrays
            else:
                out=self._numpy_price(mod,strikes,call,mi,times,div,spot,rate,yield_,smax,nx,american)
            outputs.append(out)
            self.progress.event("american_grid_prices" if american else "european_grid_prices",
                                "running",done=min(start+len(mod),len(models)),total=len(models),
                                contracts=len(strikes),backend=self.name)
        return np.concatenate(outputs,axis=0)

    @staticmethod
    def _numpy_price(models,strikes,call,mi,times,div,spot,r,q,smax,nx,american):
        grid=np.linspace(0,smax,nx);ds=grid[1]
        out=np.empty((len(models),len(strikes)))
        for b,(a0,skew) in enumerate(models):
            a=a0*(1+skew*np.tanh(np.log(np.maximum(grid/spot,1e-15))))**2
            dif=.5*a*grid*grid/(ds*ds);drift=(r-q)*grid/ds
            l=dif+np.maximum(-drift,0);u=dif+np.maximum(drift,0)
            for c,K in enumerate(strikes):
                payoff=np.maximum(grid-K if call[c] else K-grid,0);v=payoff.copy()
                last=mi[c]
                for it in range(last,0,-1):
                    if div[it]:
                        v=np.interp(np.maximum(grid-div[it],0),grid,v)
                        if american:v=np.maximum(v,payoff)
                    dt=times[it]-times[it-1];tau=times[last]-times[it-1]
                    left=0 if call[c] else K*np.exp(-r*tau)
                    right=max(smax*np.exp(-q*tau)-K*np.exp(-r*tau),0) if call[c] else 0
                    if american:left=max(left,payoff[0]);right=max(right,payoff[-1])
                    band=np.zeros((3,nx-2));band[0,1:]=-dt*u[1:-2]
                    band[1]=1+dt*(l[1:-1]+u[1:-1]+r);band[2,:-1]=-dt*l[2:-1]
                    rhs=v[1:-1].copy();rhs[0]+=dt*l[1]*left;rhs[-1]+=dt*u[-2]*right
                    middle=solve_banded((1,1),band,rhs,check_finite=False)
                    v=np.r_[left,middle,right]
                    if american:v=np.maximum(v,payoff)
                out[b,c]=np.interp(spot,grid,v)
        return out
