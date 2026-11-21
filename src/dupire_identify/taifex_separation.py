"""Verified attainable gamma averages for two-stage martingales using Arb balls."""
from fractions import Fraction as F
from math import ceil, floor

from flint import arb, ctx, fmpq

from .interval import black_call
from .spxw_books import parity, span
from .spxw_gamma import choose_reference
from .taifex_books import prepare
from .taifex_txo import check_settings, select


def ball(value):
    value = F(value)
    return arb(fmpq(value.numerator, value.denominator))


def endpoints(x):
    if not x.is_finite():
        raise ArithmeticError("Nonfinite enclosure")
    return F(str(x.lower().fmpq())), F(str(x.upper().fmpq()))


def decimal_grid(units, digits):
    w, d = divmod(abs(units), 10**digits)
    return ("-" if units < 0 else "") + f"{w}.{d:0{digits}d}"


def outward(x, digits=6):
    lo, hi = endpoints(x)
    scale = 10**digits
    return [decimal_grid(floor(lo*scale), digits), decimal_grid(ceil(hi*scale), digits)]


def cdf(z, sqrt_two):
    # erfc is decreasing. Evaluate the extremal arguments on each real interval.
    arg = -z / sqrt_two
    return (arg.lower().erfc()/2).union(arg.upper().erfc()/2)


def black(k, v, sqrt_two):
    root = v.sqrt()
    z = -k.log()/root + root/2
    return cdf(z, sqrt_two) - k*cdf(z-root, sqrt_two)


def gamma_mass(k, log_k, v, sqrt_two_pi):
    return k * (-((log_k+v/2)**2)/(2*v)).exp() / (sqrt_two_pi*v.sqrt())


def probability_profile(edges, weights):
    return (ball(weights[-1]), [(ball(edge), ball(l-r))
            for edge, l, r in zip(edges, weights[:-1], weights[1:]) if l != r])


def covered_probability(profile, mean, deviation, sqrt_two):
    last, changes = profile
    return last + sum((change*cdf((edge-mean)/deviation, sqrt_two)
                       for edge, change in changes), arb(0))


def inspect_model(model, lower, upper):
    forward, discount = F(model["forward"]), F(model["discount"])
    edges = tuple(map(F, model["edges"]))
    regimes = []
    if min(forward, discount) <= 0 or list(edges) != sorted(set(edges)):
        raise ValueError("Invalid curve or bin boundaries")
    for branch in model["regimes"]:
        fraction, initial, weight = (F(branch[k]) for k in ("time_fraction", "a0", "mass"))
        high = tuple(map(F, branch["high"]))
        if (not 0 < fraction < 1 or not lower <= initial <= upper or not 0 < weight <= 1
                or len(high) != len(edges)+1 or any(not 0 <= q <= 1 for q in high)):
            raise ValueError("Invalid regime time, variance, mixture or conditional probability")
        regimes.append((fraction, initial, weight, high))
    if sum(r[2] for r in regimes) != 1:
        raise ValueError("Regime masses must sum to one exactly")
    return forward, discount, edges, regimes


def integrals(k_value, t_value, r_value, model, lower, upper, steps):
    """Enclose integral of w, of a*w, and of (a-r)*w, in full space.

    Each ball for u covers an entire time subinterval. Conditional Gaussian
    probabilities are exact formulae for the bin weights under the gamma kernel.
    The calculation never samples a stochastic path.
    """
    _, _, edges, regimes = model
    k, t, r = map(ball, (k_value, t_value, r_value))
    log_k = k.log()
    sqrt_two, sqrt_two_pi = arb(2).sqrt(), (2*arb.pi()).sqrt()
    total_d, total_n, total_delta = arb(0), arb(0), arb(0)
    avgs = [arb(2*i+1, 1)/(2*steps) for i in range(steps)]
    for fraction, initial, weight, high in regimes:
        t0, t1 = t*ball(fraction), t*ball(1-fraction)
        a0, mass = ball(initial), ball(weight)
        initial_v = a0*t0
        first = sum((gamma_mass(k, log_k, r*t+(a0-r)*t0*u, sqrt_two_pi)
                     for u in avgs), arb(0))*(t0/steps)*mass
        total_d += first
        total_n += a0*first
        total_delta += (a0-r)*first
        for a_value, probs in ((lower, tuple(1-p for p in high)), (upper, high)):
            if not any(probs):
                continue
            a = ball(a_value)
            profile = probability_profile(edges, probs)
            accum = arb(0)
            for u in avgs:
                v = t1*(r+(a-r)*u)
                vt = initial_v+v
                mean = initial_v.sqrt()*(log_k+vt/2)/vt
                deviation = (v/vt).sqrt()
                probability = covered_probability(profile, mean, deviation, sqrt_two)
                accum += gamma_mass(k, log_k, vt, sqrt_two_pi)*probability
            segment = accum*(t1/steps)*mass
            total_d += segment
            total_n += a*segment
            total_delta += (a-r)*segment
    return total_d, total_n, total_delta


def tail_ceiling(k_value, t_value, rate, corridor):
    """Upper bound for integrated w outside the corridor, for every path law."""
    k, t, r = map(ball, (k_value, t_value, rate))
    left, right = map(ball, corridor)
    log_dist = min((k/left).log().lower(), (right/k).log().lower())
    distance = log_dist-r*t/2
    if not (distance > 0 and distance**2 >= r*t):
        raise ValueError("The small-time tail bound does not apply to this contract")
    # At fixed positive distance, exp(-distance^2/(2v))/sqrt(v) increases to v=distance^2.
    count = 64
    total = arb(0)
    for j in range(1, count+1):
        v = r*t*j/count
        total += k*(-distance**2/(2*v)).exp()/(2*arb.pi()*v).sqrt()
    return (total*t/count).upper()


def evaluate_model(spec, pairs, target, t, rates, bounds, corridor, steps):
    lo, hi = bounds
    model = inspect_model(spec, lo, hi)
    forward, disc, _, _ = model
    prices = []
    sqrt_two = arb(2).sqrt()
    for pair in pairs:
        k = pair.k/forward
        # Under the upper variance reference the price correction has only one sign.
        _, _, correction = integrals(k, t, hi, model, lo, hi, steps)
        normalized = black(ball(k), ball(hi*t), sqrt_two) + correction/2
        call = ball(disc*forward)*normalized
        put = call + ball(disc*(pair.k-forward))
        for book, price in ((pair.call, call), (pair.put, put)):
            p, q = endpoints(price)
            if not book.bid <= p <= q <= book.ask:
                raise ValueError(f"Price enclosure exceeds spread for {book.id}: {outward(price)}")
            prices.append(dict(contract=book.id, bid=str(book.bid), ask=str(book.ask),
                               price=outward(price), slack_at_least=decimal_grid(
                                   floor(min(p-book.bid, book.ask-q)*10**6), 6)))
    gamma = {}
    for name, rate in rates.items():
        k = target.k/forward
        d, n, delta = integrals(k, t, rate, model, lo, hi, steps)
        tail = tail_ceiling(k, t, rate, corridor)
        if not d.lower() > tail:
            raise ArithmeticError("No verified positive corridor occupation")
        # Both full-space formulae are valid. Delta avoids cancellation near the reference.
        full = (ball(rate)+delta/d).intersection(n/d)
        correction = (ball(hi-lo)*tail/(d.lower()-tail)).upper()
        interval = full+arb(0, correction)
        gamma[name] = dict(reference_variance=str(rate), corridor_average=outward(interval),
                           full_space_average=outward(full), tail_mass_upper=outward(tail,9)[1],
                           full_space_mass=outward(d))
    return dict(forward=str(forward), discount=str(disc), prices=prices, gamma=gamma)


def all_quotes_at_cap(pairs, t, amax):
    """A convex payoff in a bounded-volatility martingale is dominated by Black at the cap."""
    zone=parity(pairs)
    if not zone:
        return dict(pairs=len(pairs),status="empty_parity_polygon")
    forwards=span(zone,lambda d,a:a/d)
    dmax=span(zone,lambda d,a:d).hi
    failures=[]
    for pair in pairs:
        k=pair.k
        call=forwards.hi*black_call(k/forwards.hi,amax*t)
        put=forwards.lo*black_call(k/forwards.lo,amax*t)+k-forwards.lo
        for book,maximum in ((pair.call,dmax*call.hi),(pair.put,dmax*put.hi)):
            if book.bid>maximum:
                failures.append(dict(contract=book.id,bid=str(book.bid),
                                     upper_points=decimal_grid(ceil(maximum*10**6),6)))
    return dict(pairs=len(pairs),status="incompatible_with_cap" if failures else "no_violation_found",
                discount_upper=str(dmax),forward_range=list(map(str,(forwards.lo,forwards.hi))),
                contradictions=failures)


def separate(case, raw, index, steps=1024):
    if not 32 <= steps <= 65536:
        raise ValueError("Use between 32 and 65536 integration cells per phase")
    config = case["settings"]
    check_settings(config)
    pairs, spot, t, audit = prepare(raw, index, config)
    chosen, reason = select(pairs, spot, config)
    if reason or audit["panel_sha256"] != case["panel_sha256"]:
        raise ValueError("Market panel differs from the supplied case")
    target = min(chosen, key=lambda p: (abs(p.k-spot), p.k))
    if target.call.id != case["target_call"]:
        raise ValueError("Target contract differs from the supplied case")
    lo, hi = map(F, (config["amin"], config["amax"]))
    corridor = (F("0.65"), F("1.50"))
    rates = {"0.04": F("0.04"), "auto": choose_reference(parity(chosen), target, t, lo, hi)}
    with ctx.workprec(128):
        verified = {name: evaluate_model(model, chosen, target, t, rates, (lo,hi), corridor, steps)
                    for name,model in case["models"].items()}
    conclusions = {}
    for name in rates:
        ranges = {n:list(map(F, p["gamma"][name]["corridor_average"])) for n,p in verified.items()}
        low = min(ranges, key=lambda c:ranges[c][1])
        high = max(ranges, key=lambda c:ranges[c][0])
        gap = max(F(0), ranges[high][0]-ranges[low][1])
        conclusions[name] = dict(low_model=low,high_model=high,separation_at_least=str(gap),
                                 minimax_error_at_least=decimal_grid(floor(gap/2*10**6),6),
                                 rules_out_requested_error=gap>2*F(config["epsilon"]))
    return dict(date=audit["date"], expiry=audit["expiry"], target_call=target.call.id,
                selected_strikes=[str(p.k) for p in chosen], contracts=len(chosen)*2,
                panel_sha256=audit["panel_sha256"], steps_per_phase=steps,
                corridor=list(map(str,corridor)), prior=list(map(str,(lo,hi))),
                models=verified, conclusions=conclusions, all_eligible=all_quotes_at_cap(pairs,t,hi))
