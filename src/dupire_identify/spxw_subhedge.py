"""Verified static subhedges for the occupation of the Black gamma at each time.

Discovery solves a small LP. Verification only needs its proposed risky
holdings and slope. Cash is recomputed at the exact worst vertex of a
piecewise linear payoff; the semi-infinite final cell is checked separately.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import ceil
from .interval import I, exp_point, log_point, pi, sqrt


@lru_cache(maxsize=16384)
def _decay(ticks):
    """Exact lower bound on exp(-ticks/512), compressed outwards."""
    if ticks == 0:
        return F(1)
    if ticks == 1:
        return exp_point(F(-1,512)).lo
    half=_decay(ticks//2)
    value=half*half
    if ticks%2:
        value*=_decay(1)
    return I(value).rounded(80).lo


class Subhedges:
    def __init__(self, quotes, k, t, rate, corridor):
        self.calls=[(strike.hi,c.hi) for strike,c,_ in quotes]
        self.puts=[(strike.lo,p.hi) for strike,_,p in quotes]
        self.legs=[*self.calls,*self.puts]
        center=(k.lo+k.hi)/2
        width=min(center*F(3,4),5*center*sqrt(rate*t).hi)
        nodes={F(0),*(strike for strike,cost in self.legs)}
        nodes.update(center+width*F(j,160) for j in range(-160,161))
        if corridor:
            nodes.update(corridor)
        self.nodes=sorted(nodes)
        self.geometry=[]
        for l,r in zip(self.nodes,self.nodes[1:]):
            if l<=0 or (corridor and (l<corridor[0] or r>corridor[1])):
                self.geometry.append(None)
            else:
                dist=max(abs(log_point(l/k.hi).lo),abs(log_point(r/k.lo).hi))
                self.geometry.append((sqrt(k.lo*l).lo,dist**2))
        self.values=[[max(x-strike,0) for x in self.nodes] for strike,_ in self.calls]
        self.values += [[max(strike-x,0) for x in self.nodes] for strike,_ in self.puts]
        self.matrix=None

    def heights(self,v0,v1):
        if v0==0:
            return [F(0)]*len(self.nodes)
        denom=sqrt(2*pi()*v1).hi
        h=[F(0)]
        for geom in self.geometry:
            height=F(0)
            if geom:
                factor,dsq=geom
                ticks=ceil(512*(dsq/(2*v0)+v1/8))
                if ticks<24*512:
                    height=I(factor*_decay(ticks)/denom).rounded(64).lo
            h.append(height)
        return [min(a,b) for a,b in zip(h,h[1:]+[F(0)])]

    def propose(self,heights):
        import numpy as np
        from scipy.optimize import linprog
        if self.matrix is None:
            n=len(self.nodes)
            self.matrix=np.column_stack([np.ones(n),np.array(self.nodes,float)-1,
                                         *[-np.asarray(row,float) for row in self.values]])
            self.matrix=np.vstack((self.matrix,[0,1,*[-1]*len(self.calls),*[0]*len(self.puts)]))
        cost=[-1,0,*[float(p) for _,p in self.legs]]
        res=linprog(cost,A_ub=self.matrix,b_ub=[*map(float,heights),0],
                    bounds=[(None,None),(None,None)]+[(0,None)]*len(self.legs),method='highs')
        if not res.success:
            return {'slope':'0','short':{}}
        grid=2**22
        weights={str(i):str(F(round(float(v)*grid),grid)) for i,v in enumerate(res.x[2:]) if v>1/grid}
        return dict(slope=str(F(round(float(res.x[1])*grid),grid)),short=weights)

    def check(self,positions,heights):
        weights={int(i):F(w) for i,w in positions['short'].items()}
        if any(i<0 or i>=len(self.legs) or w<0 for i,w in weights.items()):
            raise ValueError('Invalid static hedge positions')
        b=min(F(positions['slope']),sum((w for i,w in weights.items() if i<len(self.calls)),F(0)))
        limits=[h-b*(x-1) for h,x in zip(heights,self.nodes)]
        for i,w in weights.items():
            limits=[v+w*payoff for v,payoff in zip(limits,self.values[i])]
        cash=min(limits)
        floor=cash-sum(w*self.legs[i][1] for i,w in weights.items())
        return max(F(0),I(floor).rounded(64).lo)
