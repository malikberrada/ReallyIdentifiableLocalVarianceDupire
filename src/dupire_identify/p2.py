"""P2 maximum-principle residual certificate for polynomial call surfaces.

Power coefficients are rational and Bernstein enclosures control the entire
rectangles, not samples. A fixed polynomial local variance may be supplied;
otherwise the guarantee is uniform for every measurable a in [amin,amax].
"""
from fractions import Fraction as F
from .interval import I,exp_point
from .polynomial import p2_deriv,p2_add,p2_mul,p2_range,p2_value
from .io import rational,quote_payload
from .progress import QUIET


def parse_poly(rows):
    p={}
    for row in rows:
        i,j=int(row[0]),int(row[1])
        if i<0 or j<0 or i!=row[0] or j!=row[1]: raise ValueError("Nonnegative integer powers required")
        p[i,j]=p.get((i,j),F(0))+rational(row[2])
    if not p: raise ValueError("Empty polynomial")
    return p


def certify_polynomial(config,quotes=(),progress=QUIET):
    v=parse_poly(config["call_polynomial"])
    amin,amax=map(rational,(config["amin"],config["amax"]))
    ell,right,tmax=map(rational,(config["kmin"],config["kmax"],config["tmax"]))
    cells=int(config.get("cells",8))
    if not 0<amin<=amax or not 0<ell<1<right or tmax<=0 or cells<1:
        raise ValueError("Invalid P2 domain or variance assumptions")
    vt=p2_deriv(v,0)
    k2vkk=p2_mul({(0,2):F(1)},p2_deriv(p2_deriv(v,1),1))
    a=parse_poly(config["variance_polynomial"]) if "variance_polynomial" in config else None
    if a:
        residuals=[p2_add(vt,{ij:-c/2 for ij,c in p2_mul(a,k2vkk).items()})]
    else:
        residuals=[p2_add(vt,{ij:-av*c/2 for ij,c in k2vkk.items()}) for av in (amin,amax)]
    rho=F(0)
    for it in range(cells):
        ta,tb=tmax*it/cells,tmax*(it+1)/cells
        for ik in range(cells):
            ka,kb=ell+(right-ell)*ik/cells,ell+(right-ell)*(ik+1)/cells
            if a:
                ar=p2_range(a,ta,tb,ka,kb)
                if ar.lo<amin or ar.hi>amax:
                    raise ValueError("Variance bounds not verified by Bernstein; subdivide or change polynomial")
            for rp in residuals:
                r=p2_range(rp,ta,tb,ka,kb)
                rho=max(rho,abs(r.lo),abs(r.hi))
        progress.event("p2_residual_boxes","running",done=it+1,total=cells)
    # Initial payoff mismatch: split at the kink k=1, then bound exactly.
    init=F(0)
    for ka,kb,payoff in [(ell,F(1),{(0,0):F(1),(0,1):F(-1)}),
                         (F(1),right,{(0,0):F(0)})]:
        p=p2_add(v,{ij:-c for ij,c in payoff.items()})
        for j in range(cells):
            r=p2_range(p,F(0),F(0),ka+(kb-ka)*j/cells,ka+(kb-ka)*(j+1)/cells)
            init=max(init,abs(r.lo),abs(r.hi))
    # True call boundary ranges follow from positivity, E X=1, E X^2<=exp(amax*T).
    # (x-R)+ <= x^2/(4R). No boundary value is silently imposed on the diffusion.
    boundary=F(0)
    tail=min(F(1),exp_point(amax*tmax).hi/(4*right))
    for k,true_range in [(ell,I(1-ell,1)),(right,I(0,tail))]:
        for j in range(cells):
            vr=p2_range(v,tmax*j/cells,tmax*(j+1)/cells,k,k)
            error=vr-true_range
            boundary=max(boundary,abs(error.lo),abs(error.hi))
    eta=max(init,boundary)+tmax*rho
    prices=[]
    for q in quotes:
        if q.t>tmax or not ell<=q.k<=right: raise ValueError("P2 quote outside certified domain")
        value=p2_value(v,q.t,q.k)
        prices.append(dict(id=q.id,numerical_value=str(value),eta=str(eta),
                           continuous_price=I(value-eta,value+eta).json(),
                           compatible_proved=q.bid+eta<=value<=q.ask-eta))
    return dict(scope="fixed_variance_on_domain_bounded_exterior" if a else "all_bounded_variance_continuous_diffusions",
                status="rational_maximum_principle_residual_bound",eta=str(eta),
                approximate_eta=float(eta),initial_error=str(init),
                boundary_and_tail_error=str(boundary),residual_sup=str(rho),
                arithmetic="exact rational Bernstein bounds", prices=prices,
                all_quotes_compatible_proved=all(p["compatible_proved"] for p in prices) if prices else None,
                config=config,quotes=quote_payload(quotes),
                optimization_error="Candidate supplied as exact rational coefficients; no solver tolerance discarded",
                theorem="|c-v| <= max(initial_error,boundary_error)+T*residual_sup",
                limitation="A valid bound can be wide; no universal shrinking-rate or sharpness claim")
