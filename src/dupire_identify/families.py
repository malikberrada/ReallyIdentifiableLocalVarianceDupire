"""Closed families of positive kernels, not sampled parameter grids."""
from dataclasses import dataclass
from fractions import Fraction as F
from .continuous import Window
from .interval import I
from .polynomial import bump,bernstein_range


def square(x):
    return I(0 if x.lo<=0<=x.hi else min(x.lo*x.lo,x.hi*x.hi),max(x.lo*x.lo,x.hi*x.hi))


def bump_range(u):
    q=square(u)
    return I(max(F(0),1-q.hi)**3,max(F(0),1-q.lo)**3)


@dataclass(frozen=True)
class Family:
    coordinates: str
    intervals: tuple

    @classmethod
    def make(cls,doc):
        kind=doc.get('coordinates','center')
        keys=('t','k','ht','hk') if kind=='center' else ('u','v','ht','hk') if kind=='left' else ()
        if not keys or set(doc)-{'coordinates'}!=set(keys):raise ValueError('Use center t,k,ht,hk or left u,v,ht,hk coordinates')
        pairs=tuple(tuple(map(F,doc[k])) for k in keys)
        if any(len(p)!=2 or p[0]>p[1] for p in pairs):raise ValueError('Invalid closed parameter intervals')
        if min(pairs[2][0],pairs[3][0])<=0:raise ValueError('Positive halfwidths required')
        if kind=='center':
            if pairs[0][0]<=pairs[2][1] or pairs[1][0]<=pairs[3][1]:raise ValueError('Entire family must have positive supports')
        elif min(pairs[0][0],pairs[1][0])<=0:raise ValueError('Positive left supports required')
        return cls(kind,pairs)

    @classmethod
    def point(cls,w):return cls.make({k:[v,v] for k,v in w.json().items()})

    def json(self):
        keys=('t','k','ht','hk') if self.coordinates=='center' else ('u','v','ht','hk')
        return dict(coordinates=self.coordinates,**{k:list(map(str,p)) for k,p in zip(keys,self.intervals)})

    def center(self):
        c=[sum(p)/2 for p in self.intervals]
        if self.coordinates=='left':c[0]+=c[2];c[1]+=c[3]
        return Window.make(*c)

    def support(self):
        a,b,h,j=self.intervals
        if self.coordinates=='center':return a[0]-h[1],a[1]+h[1],b[0]-j[1],b[1]+j[1]
        return a[0],a[1]+2*h[1],b[0],b[1]+2*j[1]

    def cover_window(self):
        ta,tb,ka,kb=self.support()
        return Window.make((ta+tb)/2,(ka+kb)/2,(tb-ta)/2,(kb-ka)/2)

    def weights(self,ta,tb,ka,kb):
        if all(a==b for a,b in self.intervals):
            w=self.center()
            # Cells of the physical relaxation partition the full support.
            if tb<=w.t-w.ht or ta>=w.t+w.ht or kb<=w.k-w.hk or ka>=w.k+w.hk:return I(0)
            if ta>=w.t-w.ht and tb<=w.t+w.ht and ka>=w.k-w.hk and kb<=w.k+w.hk:
                pt,pk=bump(w.t,w.ht),bump(w.k,w.hk)
                z=bernstein_range(pt,ta,tb)*bernstein_range(pk,ka,kb)
                return I(max(F(0),z.lo),min(F(1),max(F(0),z.hi)))
        a,b,h,j=[I(*p) for p in self.intervals]
        ut=(I(ta,tb)-a)/h;uk=(I(ka,kb)-b)/j
        if self.coordinates=='left':ut=ut-1;uk=uk-1
        return bump_range(ut)*bump_range(uk)

    def common_kernel(self):
        """phi_family >= c*phi_reference, by a convex-square inequality."""
        a,b,h,j=self.intervals
        centers=[a,b]
        if self.coordinates=='left':centers=[(a[0]+h[0],a[1]+h[1]),(b[0]+j[0],b[1]+j[1])]
        cc=[];half=[];factor=F(1)
        for center,width in zip(centers,(h,j)):
            mid=sum(center)/2;delta=(center[1]-center[0])/2
            if delta>=width[0]:return None
            radius=F(9,10)*(width[0]-delta);alpha=delta/width[0];beta=radius/width[0]
            c=1-alpha*alpha/(1-beta)
            if c<=0:return None
            factor*=c**3;cc.append(mid);half.append(radius)
        return Window.make(cc[0],cc[1],half[0],half[1]),factor

    def weight_deviation(self):
        """Uniform |phi_w-phi_center|, using |bump'|<2 on the line."""
        r=[(p[1]-p[0])/2 for p in self.intervals];h,j=self.intervals[2:]
        extra=2 if self.coordinates=='left' else 1
        return min(F(1),2*(r[0]+extra*r[2])/h[0]+2*(r[1]+extra*r[3])/j[0])

    def split(self,axis=None,middle=None):
        widths=[(b-a)/max(F(1,100),abs((a+b)/2)) for a,b in self.intervals]
        axis=max(range(4),key=lambda i:widths[i]) if axis is None else axis
        if axis not in range(4):raise ValueError('Invalid split axis')
        a,b=self.intervals[axis]
        if a==b:return None
        mid=(a+b)/2 if middle is None else F(middle)
        if not a<mid<b:raise ValueError('Split point must be strictly interior')
        left=list(self.intervals);right=list(self.intervals)
        left[axis]=(a,mid);right[axis]=(mid,b)
        return axis,mid,Family(self.coordinates,tuple(left)),Family(self.coordinates,tuple(right))


def shell(index):
    if index<0:raise ValueError('Nonnegative shell index required')
    lo,hi=F(1,2**index),F(2**index)
    return Family.make(dict(coordinates='left',**{k:[str(lo),str(hi)] for k in ('u','v','ht','hk')}))
