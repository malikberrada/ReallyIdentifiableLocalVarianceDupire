// Binary64 SDIRK2 proposals, one parallel cyclic-reduction block per model.
// No host headers, fast math, atomic reductions or certificate calculations.
extern "C" __global__ void fd_batch_kernel(
    int Bn, int P, int N, int levels, const double* grid,
    const double* coeff, const double* lefts, const double* rights,
    const int* counts, const double* initial, const double* window,
    double* factors, double* curves, double* nums, double* dens)
{
    const int model=blockIdx.x, j=threadIdx.x, M=N-2, S=blockDim.x;
    if(model>=Bn) return; // Uniform across the complete block.
    extern __shared__ double sh[];
    double* c=sh;
    double* y=c+S+2;
    double* r=y+S+2;
    double* aux=r+S;
    const bool active=j<M;
    const double gamma=1.0-1.0/sqrt(2.0);
    double ql=0.0,qr=0.0,qd=0.0,x=0.0,weight=0.0,pk=0.0;
    if(active){
        double hl=grid[j+1]-grid[j], hr=grid[j+2]-grid[j+1];
        ql=2.0/(hl*(hl+hr)); qr=2.0/(hr*(hl+hr)); qd=-ql-qr;
        x=grid[j+1]; weight=(hl+hr)/2.0;
        double u=(x-window[1])/window[3];
        double z=fmax(0.0,1.0-u*u); pk=z*z*z;
        c[j+1]=initial[model*N+j+1];
    }
    if(j==0){c[0]=initial[model*N];c[M+1]=initial[model*N+N-1];}
    __syncthreads();
    double num=0.0,den=0.0;
    double* fact=factors+((long long)model)*2*levels*M;
    for(int p=0;p<P;++p){
        int count=counts[p]; double dt=(rights[p]-lefts[p])/count;
        double variance=active?coeff[((long long)model*P+p)*N+j+1]:0.0;
        double D=0.5*variance*x*x, L=D*ql, G=D*qd, U=D*qr;
        double a=active&&j>0?-gamma*dt*L:0.0;
        double b=active?1.0-gamma*dt*G:1.0;
        double d=active&&j<M-1?-gamma*dt*U:0.0;
        // Reuse y, r and aux for the constant-matrix factorisation.
        for(int l=0,stride=1;l<levels;++l,stride*=2){
            y[j]=a;r[j]=b;aux[j]=d;
            __syncthreads();
            double alpha=active&&j>=stride?-a/r[j-stride]:0.0;
            double beta=active&&j+stride<M?-d/r[j+stride]:0.0;
            double aa=0.0,bb=b,dd=0.0;
            if(active&&j>=stride){aa=alpha*y[j-stride];bb+=alpha*aux[j-stride];}
            if(active&&j+stride<M){dd=beta*aux[j+stride];bb+=beta*y[j+stride];}
            if(active){fact[((long long)l*2)*M+j]=alpha;fact[((long long)l*2+1)*M+j]=beta;}
            __syncthreads();
            a=aa;b=bb;d=dd;
        }
        const double diagonal=b;
        for(int it=0;it<count;++it){
            double rhs=active?c[j+1]:0.0;
            if(active&&j==0)rhs+=gamma*dt*L*c[0];
            if(active&&j==M-1)rhs+=gamma*dt*U*c[M+1];
            r[j]=rhs;
            __syncthreads();
            for(int l=0,stride=1;l<levels;++l,stride*=2){
                double v=r[j];
                if(active&&j>=stride)v+=fact[((long long)l*2)*M+j]*r[j-stride];
                if(active&&j+stride<M)v+=fact[((long long)l*2+1)*M+j]*r[j+stride];
                __syncthreads();r[j]=v;__syncthreads();
            }
            if(active)y[j+1]=r[j]/diagonal;
            if(j==0){y[0]=c[0];y[M+1]=c[M+1];}
            __syncthreads();
            rhs=active?c[j+1]+(1.0-gamma)*dt*(L*y[j]+G*y[j+1]+U*y[j+2]):0.0;
            if(active&&j==0)rhs+=gamma*dt*L*c[0];
            if(active&&j==M-1)rhs+=gamma*dt*U*c[M+1];
            r[j]=rhs;
            __syncthreads();
            for(int l=0,stride=1;l<levels;++l,stride*=2){
                double v=r[j];
                if(active&&j>=stride)v+=fact[((long long)l*2)*M+j]*r[j-stride];
                if(active&&j+stride<M)v+=fact[((long long)l*2+1)*M+j]*r[j+stride];
                __syncthreads();r[j]=v;__syncthreads();
            }
            double value=active?r[j]/diagonal:0.0;
            if(active)y[j+1]=(c[j+1]+value)/2.0;
            __syncthreads();
            if(active){
                double mid=lefts[p]+(it+0.5)*dt, u=(mid-window[0])/window[2];
                double z=fmax(0.0,1.0-u*u), pt=z*z*z;
                double density=fmax(0.0,ql*y[j]+qd*y[j+1]+qr*y[j+2]);
                double mass=dt*pt*pk*x*x*density*weight;
                den+=mass;num+=mass*variance;
                c[j+1]=value;
            }
            __syncthreads();
        }
        if(active)curves[((long long)model*P+p)*N+j+1]=c[j+1];
        if(j==0){curves[((long long)model*P+p)*N]=c[0];curves[((long long)model*P+p)*N+N-1]=c[M+1];}
        __syncthreads();
    }
    r[j]=num;aux[j]=den;__syncthreads();
    for(int stride=S/2;stride>0;stride/=2){
        if(j<stride){r[j]+=r[j+stride];aux[j]+=aux[j+stride];}
        __syncthreads();
    }
    if(j==0){nums[model]=r[0];dens[model]=aux[0];}
}
