// Shared numerical pricing implementation: C++/OpenMP and CUDA/NVRTC.
// This prices a Bermudan exercise grid. It is NOT a continuous-American certificate.
#pragma once
#ifndef __CUDACC_RTC__
// Host C++/nvcc needs the header. NVRTC already provides the device math
// functions and must not depend on a host SDK or system math.h (Windows).
#include <math.h>
#endif
#ifdef __CUDACC__
#define DI_HD __device__ __forceinline__
#else
#define DI_HD inline
#endif

DI_HD double di_max(double a, double b) { return a>b?a:b; }
DI_HD double di_min(double a, double b) { return a<b?a:b; }
DI_HD double di_payoff(double s,double k,int call) {
    return di_max(call?s-k:k-s,0.0);
}

struct di_vector {
    double* p;
    long long stride;
    DI_HD double& operator[](int j) { return p[(long long)j*stride]; }
};

DI_HD void di_price_one(
    int job, const double* models, int contracts, const double* strikes,
    const int* iscall, const int* maturity_index, const double* times,
    const double* dividends, double spot, double rate, double yield,
    double smax, int nx, int american, double* work, long long workspace_stride,
    double* output
) {
    const int m=job/contracts,c=job%contracts;
    const double a0=models[2*m],skew=models[2*m+1],K=strikes[c],ds=smax/(nx-1);
    // GPU: structure-of-arrays, coalesced across contracts/models at each j.
    // CPU: stride=1 and one local work vector per OpenMP iteration.
    const long long offset=workspace_stride==1?0:job;
    di_vector v{work+offset,workspace_stride};
    di_vector old{work+offset+(long long)nx*workspace_stride,workspace_stride};
    di_vector cp{work+offset+(long long)nx*workspace_stride*2,workspace_stride};
    di_vector dp{work+offset+(long long)nx*workspace_stride*3,workspace_stride};
    di_vector lc{work+offset+(long long)nx*workspace_stride*4,workspace_stride};
    di_vector uc{work+offset+(long long)nx*workspace_stride*5,workspace_stride};
    const int last=maturity_index[c];
    for(int j=0;j<nx;j++) {
        v[j]=di_payoff(j*ds,K,iscall[c]);
        const double s=j*ds;
        const double shape=1+skew*tanh(log(di_max(s/spot,1e-15)));
        const double diffusion=0.5*a0*shape*shape*s*s/(ds*ds);
        const double drift=(rate-yield)*s/ds;
        lc[j]=diffusion+di_max(-drift,0.0);
        uc[j]=diffusion+di_max(drift,0.0);
    }
    for(int it=last;it>0;it--) {
        // Jump at the right endpoint: V(t-,S)=V(t+,max(S-D,0)).
        // Exercise is available just before the jump on the Bermudan grid.
        const double div=dividends[it];
        if(div>0.0) {
            for(int j=0;j<nx;j++) old[j]=v[j];
            for(int j=0;j<nx;j++) {
                const double x=di_max(j*ds-div,0.0)/ds;
                int l=(int)floor(x); if(l>nx-2) l=nx-2;
                const double z=x-l;
                v[j]=old[l]*(1-z)+old[l+1]*z;
                if(american) v[j]=di_max(v[j],di_payoff(j*ds,K,iscall[c]));
            }
        }
        for(int j=0;j<nx;j++) old[j]=v[j];
        const double dt=times[it]-times[it-1],tau=times[last]-times[it-1];
        // Approximate finite-domain Dirichlet values (reported as numerical).
        double left=iscall[c]?0.0:K*exp(-rate*tau);
        double right=iscall[c]?di_max(smax*exp(-yield*tau)-K*exp(-rate*tau),0.0):0.0;
        if(american) {
            left=di_max(left,di_payoff(0,K,iscall[c]));
            right=di_max(right,di_payoff(smax,K,iscall[c]));
        }
        v[0]=left;v[nx-1]=right;
        cp[0]=0;dp[0]=left;
        for(int j=1;j<nx-1;j++) {
            const double l=lc[j],u=uc[j];
            const double lower=-dt*l,upper=-dt*u,diag=1+dt*(l+u+rate);
            double rhs=old[j];
            if(j==nx-2) rhs-=upper*right;
            const double pivot=diag-lower*cp[j-1];
            cp[j]=j==nx-2?0.0:upper/pivot;
            dp[j]=(rhs-lower*dp[j-1])/pivot;
        }
        for(int j=nx-2;j>=1;j--) v[j]=dp[j]-cp[j]*v[j+1];
        if(american) for(int j=0;j<nx;j++)
            v[j]=di_max(v[j],di_payoff(j*ds,K,iscall[c]));
    }
    double x=spot/ds; int j=(int)floor(x);if(j>nx-2)j=nx-2;
    output[job]=(1-(x-j))*v[j]+(x-j)*v[j+1];
}
