"""Evaluate generated continuous models directly from their certificates."""
from fractions import Fraction as F
from .interval import I,black_call
from .inner import basis_values
from .density import gamma_box
from . import time_models


class ContinuousModel:
    """Call verify-tighten before using a model certificate from an untrusted file."""
    def __init__(self,certificate):
        self.doc=certificate
        self.time_change=certificate.get('model_type')=='deterministic_time_change'
        if not self.time_change:
            self.t=tuple(map(F,certificate['definition']['time_knots']))
            self.k=tuple(map(F,certificate['definition']['strike_knots']))
            self.theta=list(map(F,certificate['theta']))
            self.variances=list(map(F,certificate['baseline']['variances']))
            self.weights=list(map(F,certificate['baseline']['weights']))

    def correction(self,t,k,dt=0,dk=0):
        if self.time_change:return F(0)
        bt=basis_values(self.t,t,dt);bk=basis_values(self.k,k,dk)
        return sum(a*b*self.theta[i*len(bk)+j] for i,a in enumerate(bt) for j,b in enumerate(bk) if a and b)

    def call(self,t,k):
        t,k=F(str(t)),F(str(k))
        if t<0 or k<0:raise ValueError('Nonnegative time and strike required')
        if k==0:return I(1)
        if self.time_change:return black_call(k,time_models.integrated_variance(self.doc,t))
        return (sum((w*black_call(k,a*t) for w,a in zip(self.weights,self.variances)),I(0))+
                self.correction(t,k)).rounded()

    def local_variance(self,t,k):
        t,k=F(str(t)),F(str(k))
        if t<=0 or k<=0:raise ValueError('Positive time and state required')
        if self.time_change:return I(time_models.variance(self.doc,t))
        prior=I(F(self.doc['amin']),F(self.doc['amax']))
        gs=[gamma_box(t,t,k,k,a)[0] for a in self.variances]
        denominator=sum((w*g for w,g in zip(self.weights,gs)),I(0))+k*k*self.correction(t,k,dk=2)
        numerator=sum((w*a*g for w,a,g in zip(self.weights,self.variances,gs)),I(0))+2*self.correction(t,k,dt=1)
        if denominator.lo<=0:return prior
        return (numerator/denominator).rounded().intersect(prior)
