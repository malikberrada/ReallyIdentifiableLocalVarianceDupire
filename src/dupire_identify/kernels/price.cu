#include "pricer.h"
extern "C" __global__ void price_kernel(
    int jobs,const double* models,int contracts,const double* strikes,
    const int* iscall,const int* maturity_index,const double* times,
    const double* dividends,double spot,double rate,double yield,
    double smax,int nx,int american,double* work,double* output
) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<jobs) di_price_one(j,models,contracts,strikes,iscall,maturity_index,
                           times,dividends,spot,rate,yield,smax,nx,american,work,jobs,output);
}

extern "C" __global__ void black_kernel(
    long long jobs,int contracts,const double* strikes,const double* variances,double* out
) {
    long long j=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=jobs)return;
    int c=j%contracts,m=j/contracts;double k=strikes[c],v=variances[m];
    if(v<=0) out[j]=di_max(1-k,0.0);
    else {
        double s=sqrt(v),d1=(-log(k)+v/2)/s,d2=d1-s;
        out[j]=.5*erfc(-d1/sqrt(2.0))-k*.5*erfc(-d2/sqrt(2.0));
    }
}
