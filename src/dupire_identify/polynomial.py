"""Exact power polynomials and Bernstein range enclosures."""
from fractions import Fraction as F
from math import comb
from .interval import I


def trim(p):
    p = list(p)
    while len(p) > 1 and p[-1] == 0: p.pop()
    return p


def add(p, q):
    return trim([(p[i] if i < len(p) else 0)+(q[i] if i < len(q) else 0)
                 for i in range(max(len(p), len(q)))])


def mul(p, q):
    out = [F(0)]*(len(p)+len(q)-1)
    for i, a in enumerate(p):
        for j, b in enumerate(q): out[i+j] += a*b
    return trim(out)


def deriv(p): return [i*p[i] for i in range(1, len(p))] or [F(0)]


def integrate(p, lo, hi):
    return sum(c*(hi**(i+1)-lo**(i+1))/(i+1) for i, c in enumerate(p))


def compose_affine(p, lo, width):
    q = [F(0)]*len(p)
    for i, c in enumerate(p):
        for j in range(i+1): q[j] += c*comb(i, j)*lo**(i-j)*width**j
    return trim(q)


def bernstein_range(p, lo, hi):
    q = compose_affine(p, lo, hi-lo)
    n = len(q)-1
    b = [sum(q[j]*F(comb(i, j), comb(n, j)) for j in range(i+1)) for i in range(n+1)]
    return I(min(b), max(b))


def bump(center, halfwidth):
    if halfwidth <= 0: raise ValueError("Kernel half-width must be positive")
    u = [-center/halfwidth, 1/halfwidth]
    p = add([F(1)], [-x for x in mul(u, u)])
    return mul(mul(p, p), p)


def p2_deriv(p, axis):
    out = {}
    for (i, j), c in p.items():
        n = i if axis == 0 else j
        if n:
            ij = (i-1, j) if axis == 0 else (i, j-1)
            out[ij] = n*c
    return out or {(0, 0): F(0)}


def p2_add(p, q):
    r = dict(p)
    for ij, c in q.items(): r[ij] = r.get(ij, F(0))+c
    return {ij:c for ij,c in r.items() if c} or {(0,0):F(0)}


def p2_mul(p, q):
    out = {}
    for (i, j), a in p.items():
        for (k, l), b in q.items():
            out[i+k, j+l] = out.get((i+k, j+l), F(0))+a*b
    return out


def p2_range(p, ta, tb, ka, kb):
    nt = max(i for i, j in p)
    nk = max(j for i, j in p)
    power = {}
    for (i, j), a in p.items():
        for m in range(i+1):
            for n in range(j+1):
                c = a*comb(i,m)*ta**(i-m)*(tb-ta)**m*comb(j,n)*ka**(j-n)*(kb-ka)**n
                power[m,n] = power.get((m,n), F(0))+c
    vals = []
    for i in range(nt+1):
        for j in range(nk+1):
            vals.append(sum(c*F(comb(i,m),comb(nt,m))*F(comb(j,n),comb(nk,n))
                            for (m,n),c in power.items() if m<=i and n<=j))
    return I(min(vals), max(vals))


def p2_value(p, t, k):
    return sum(c*t**i*k**j for (i,j),c in p.items())
