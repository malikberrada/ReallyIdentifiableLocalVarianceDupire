#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <vector>
#include "pricer.h"
#ifdef _OPENMP
#include <omp.h>
#endif
namespace py=pybind11;
using DArray=py::array_t<double,py::array::c_style|py::array::forcecast>;
using IArray=py::array_t<int,py::array::c_style|py::array::forcecast>;

DArray black(DArray ks,DArray vs,int threads) {
    auto k=ks.unchecked<1>();auto v=vs.unchecked<1>();
    const long long B=v.shape(0),N=k.shape(0);
    DArray out({B,N});auto o=out.mutable_unchecked<2>();
    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(long long b=0;b<B;b++) for(long long j=0;j<N;j++) {
        if(v(b)<=0) o(b,j)=di_max(1-k(j),0.0);
        else {
            double s=sqrt(v(b)),d1=(-log(k(j))+.5*v(b))/s,d2=d1-s;
            o(b,j)=.5*erfc(-d1/sqrt(2.0))-k(j)*.5*erfc(-d2/sqrt(2.0));
        }
    }
    return out;
}

DArray price(DArray models,DArray strikes,IArray call,IArray maturities,
             DArray times,DArray dividends,double spot,double rate,double yield,
             double smax,int nx,bool american,int threads) {
    const auto n=models.shape(0)*strikes.size();
    DArray out({models.shape(0),strikes.size()});
    const double *mp=models.data(),*kp=strikes.data(),*tp=times.data(),*dp=dividends.data();
    const int *cp=call.data(),*mi=maturities.data(); double* op=out.mutable_data();
    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(long long j=0;j<n;j++) {
        // One small per-thread work vector, avoiding a B*N*nx allocation on CPU.
        double* work=new double[(long long)nx*6];
        // Shift job's workspace indexing back to zero, while retaining its inputs.
        const int m=(int)(j/strikes.size()),c=(int)(j%strikes.size());
        di_price_one(0,mp+2*m,1,kp+c,cp+c,mi+c,tp,dp,spot,rate,yield,
                     smax,nx,american,work,1,op+j);
        delete[] work;
    }
    return out;
}

DArray tridiagonal(DArray matrix,DArray rhs) {
    auto a=matrix.unchecked<2>();auto r=rhs.unchecked<1>();
    if(a.shape(0)!=3 || a.shape(1)!=r.shape(0) || r.shape(0)<1)
        throw std::invalid_argument("Invalid tridiagonal dimensions");
    const auto n=r.shape(0);DArray out(n);auto x=out.mutable_unchecked<1>();
    std::vector<double> d(n);
    py::gil_scoped_release release;
    for(py::ssize_t i=0;i<n;i++){d[i]=a(1,i);x(i)=r(i);}
    for(py::ssize_t i=1;i<n;i++){
        if(d[i-1]==0)throw std::runtime_error("Zero tridiagonal pivot");
        const double factor=a(2,i-1)/d[i-1];
        d[i]-=factor*a(0,i);x(i)-=factor*x(i-1);
    }
    if(d[n-1]==0)throw std::runtime_error("Zero tridiagonal pivot");
    x(n-1)/=d[n-1];
    for(py::ssize_t i=n-1;i-->0;){
        if(d[i]==0)throw std::runtime_error("Zero tridiagonal pivot");
        x(i)=(x(i)-a(0,i+1)*x(i+1))/d[i];
    }
    return out;
}

py::tuple fd_batch(DArray grid,DArray coefficients,DArray lefts,DArray rights,
                   DArray initial,int steps,DArray window,int threads) {
    auto x=grid.unchecked<1>();auto a=coefficients.unchecked<3>();
    auto ls=lefts.unchecked<1>();auto rs=rights.unchecked<1>();auto init=initial.unchecked<2>();auto w=window.unchecked<1>();
    const py::ssize_t B=a.shape(0),P=a.shape(1),N=x.shape(0),M=N-2;
    if(N<3 || a.shape(2)!=N || init.shape(0)!=B || init.shape(1)!=N || ls.shape(0)!=P || rs.shape(0)!=P || w.shape(0)!=4 || steps<1 || threads<1)
        throw std::invalid_argument("Invalid batched FD dimensions");
    for(py::ssize_t i=0;i<N;i++)if(!std::isfinite(x(i)) || x(i)<=0 || (i && x(i)<=x(i-1)))throw std::invalid_argument("Invalid FD grid");
    for(py::ssize_t p=0;p<P;p++)if(!(rs(p)>ls(p)) || (rs(p)-ls(p))*steps>1e7)throw std::invalid_argument("FD phase exceeds budget");
    for(py::ssize_t b=0;b<B;b++)for(py::ssize_t p=0;p<P;p++)for(py::ssize_t i=0;i<N;i++)
        if(!std::isfinite(a(b,p,i)*x(i)*x(i)) || a(b,p,i)<=0)throw std::invalid_argument("Nonpositive or nonfinite FD coefficient");
    DArray curves({B,P,N});auto out=curves.mutable_unchecked<3>();DArray num(B),den(B);auto ns=num.mutable_unchecked<1>();auto ds=den.mutable_unchecked<1>();
    std::vector<double> ql(M),qr(M),qd(M),quad(N),phi(N);
    for(py::ssize_t i=0;i<M;i++){
        const double l=x(i+1)-x(i),r=x(i+2)-x(i+1);
        ql[i]=2/(l*(l+r));qr[i]=2/(r*(l+r));qd[i]=-ql[i]-qr[i];
    }
    for(py::ssize_t i=0;i<N;i++){
        quad[i]=(i==0?(x(1)-x(0)):(i==N-1?(x(N-1)-x(N-2)):(x(i+1)-x(i-1))))/2;
        const double u=(x(i)-w(1))/w(3);const double v=std::abs(u)<1?1-u*u:0.;phi[i]=v*v*v*x(i)*x(i)*quad[i];
    }
    {
        py::gil_scoped_release release;
        #pragma omp parallel for num_threads(threads) schedule(static)
        for(py::ssize_t model=0;model<B;model++){
            std::vector<double> c(N),old(N),y(N),L(M),D(M),U(M),lower(M),diag(M),upper(M),rhs(M),factor(M),mult(M);
            for(py::ssize_t i=0;i<N;i++)c[i]=init(model,i);
            ns(model)=0;ds(model)=0;const double gamma=1-1/std::sqrt(2.);
            for(py::ssize_t p=0;p<P;p++){
                const int count=std::max(1,(int)std::ceil((rs(p)-ls(p))*steps));const double dt=(rs(p)-ls(p))/count;
                for(py::ssize_t i=0;i<M;i++){
                    const double d=.5*a(model,p,i+1)*x(i+1)*x(i+1);
                    L[i]=d*ql[i];D[i]=d*qd[i];U[i]=d*qr[i];
                    lower[i]=-gamma*dt*L[i];diag[i]=1-gamma*dt*D[i];upper[i]=-gamma*dt*U[i];
                }
                factor[0]=diag[0];
                for(py::ssize_t i=1;i<M;i++){mult[i]=lower[i]/factor[i-1];factor[i]=diag[i]-mult[i]*upper[i-1];}
                auto solve=[&](std::vector<double>& b,std::vector<double>& target){
                    for(py::ssize_t i=1;i<M;i++)b[i]-=mult[i]*b[i-1];
                    target[N-2]=b[M-1]/factor[M-1];
                    for(py::ssize_t i=M-1;i-->0;)target[i+1]=(b[i]-upper[i]*target[i+2])/factor[i];
                };
                for(int step=0;step<count;step++){
                    old=c;y[0]=c[0];y[N-1]=c[N-1];
                    for(py::ssize_t i=0;i<M;i++)rhs[i]=c[i+1];
                    rhs[0]+=gamma*dt*L[0]*c[0];rhs[M-1]+=gamma*dt*U[M-1]*c[N-1];solve(rhs,y);
                    for(py::ssize_t i=0;i<M;i++)rhs[i]=c[i+1]+(1-gamma)*dt*(L[i]*y[i]+D[i]*y[i+1]+U[i]*y[i+2]);
                    rhs[0]+=gamma*dt*L[0]*c[0];rhs[M-1]+=gamma*dt*U[M-1]*c[N-1];solve(rhs,c);
                    const double u=(ls(p)+(step+.5)*dt-w(0))/w(2),v=std::abs(u)<1?1-u*u:0.,pt=v*v*v;
                    if(pt)for(py::ssize_t i=0;i<M;i++){
                        const double rho=std::max(0.,.5*(ql[i]*(old[i]+c[i])+qd[i]*(old[i+1]+c[i+1])+qr[i]*(old[i+2]+c[i+2])));
                        const double mass=dt*pt*phi[i+1]*rho;ds(model)+=mass;ns(model)+=mass*a(model,p,i+1);
                    }
                }
                for(py::ssize_t i=0;i<N;i++)out(model,p,i)=c[i];
            }
        }
    }
    return py::make_tuple(curves,num,den);
}

PYBIND11_MODULE(_native,m) {
    m.def("black",&black);
    m.def("price",&price);
    m.def("tridiagonal",&tridiagonal);
    m.def("fd_batch",&fd_batch);
    m.def("openmp",[](){
        #ifdef _OPENMP
        return true;
        #else
        return false;
        #endif
    });
}
