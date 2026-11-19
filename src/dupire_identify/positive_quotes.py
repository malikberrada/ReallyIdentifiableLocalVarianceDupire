"""Uniform bounds for the beta-window average from normalized European calls."""
from fractions import Fraction as F
from math import ceil, floor

from . import fixed_model
from .families import bump_range
from .interval import I, black_call, exp_point, log_point, pi, sqrt
from .io import quote_payload


def check_witness(quotes, amin, amax, variance, model_run, suggested=None):
    if model_run is not None:
        fixed_model.verify(model_run)
        model = fixed_model.Model(model_run)
        if (quote_payload(quotes) != quote_payload(model.quotes)
                or (amin, amax) != (F(model.config['amin']), F(model.config['amax']))):
            raise ValueError("Le modèle témoin doit porter sur les mêmes cotations et bornes de variance.")
        return {"type": "diffusion_verifiee"}

    def first_miss(rate):
        for q in quotes:
            price = black_call(q.k, rate * q.t)
            if not q.bid <= price.lo <= price.hi <= q.ask:
                return q.id
        return None

    if variance is None:
        for rate in dict.fromkeys((suggested, amax, amin)):
            if rate is not None and amin <= rate <= amax and first_miss(rate) is None:
                return {"type": "variance_constante", "variance": str(rate)}
        from .clock_models import safe_variance_ranges
        ranges = safe_variance_ranges(quotes, amin, amax) if amin < amax else None
        if ranges:
            low = max(v[0] / t for t, v in ranges.items())
            high = min(v[1] / t for t, v in ranges.items())
            if low <= high:
                variance = (low + high) / 2
        if variance is None:
            raise ValueError("Aucun témoin constant trouvé. Fournir un modèle compatible avec --modele-temoin.")
    if not amin <= variance <= amax:
        raise ValueError("La variance du témoin doit appartenir à la classe demandée.")
    miss = first_miss(variance)
    if miss is not None:
        raise ValueError(f"Témoin de variance constante hors spread pour {miss}.")
    return {"type": "variance_constante", "variance": str(variance)}


def axis_weight(family, band, position, width):
    arg = (I(*band) - I(*position)) / I(*width)
    if family.coordinates == "left":
        arg = arg - 1
    return bump_range(arg)


def denominator_bound(family, amax):
    """Uniform moment bound on the occupation of a central price band."""
    t, k, ht, hk = family.intervals
    time_mid = sum(t) / 2
    if family.coordinates == "left":
        time_mid += sum(ht) / 2
    support_lo, support_hi, _, _ = family.support()
    best, band = F(0), None
    for portion in (F(1, 4), F(1, 2), F(3, 4)):
        time = (max(support_lo, time_mid - portion * ht[0]),
                min(support_hi, time_mid + portion * ht[0]))
        time_weight = axis_weight(family, time, t, ht).lo
        if time_weight <= 0 or amax * time[1] >= 1:
            # With d<1, the second-moment probability bound is then trivial.
            continue
        second_moment = exp_point(amax * time[1]).hi - 1
        for part in (F(1, 2), F(2, 3), F(3, 4), F(4, 5), F(7, 8)):
            d = min(F(1), hk[0]) * part
            mass = 1 - second_moment / d**2
            if mass <= 0:
                continue
            stock = (1 - d, 1 + d)
            weight = axis_weight(family, stock, k, hk).lo
            bound = (time[1] - time[0]) * time_weight * weight * stock[0]**2 * mass
            if bound > best:
                best = bound
                band = {"temps": list(map(str, time)), "prix": list(map(str, stock)),
                        "probabilite_au_moins": str(mass)}
    return best, band


def gamma_lower(q, rate, time_lo, time_hi, left, right):
    variance = I(rate * (q.t - time_hi), rate * (q.t - time_lo))
    root = sqrt(variance)
    if root.lo <= 0:
        return F(0)
    log_ratio = I(log_point(left / q.k).lo, log_point(right / q.k).hi)
    d1 = (log_ratio + variance / 2) / root
    energy = max(abs(d1.lo), abs(d1.hi))**2 / 2
    # Round the exponent up before calling the project's rational exp engine.
    energy = F(ceil(16 * energy), 16)
    if energy > 100:  # Outside that engine's range, zero is a valid minorant.
        return F(0)
    return exp_point(-energy).lo / (sqrt(2 * pi()).hi * right * root.hi)


def budget_bound(quotes, family, rate, at_cap, cells):
    """Superpose call gammas that dominate each spatial beta weight."""
    t0, t1, s0, s1 = family.support()
    options = []
    for q in quotes:
        if q.t <= t1:
            continue
        price = black_call(q.k, rate * q.t)
        slack = price.hi - q.bid if at_cap else q.ask - price.lo
        if slack < 0:
            raise ValueError(f"Cotation incompatible avec la comparaison Black pour {q.id}.")
        options.append((q, slack))
    selected = {}
    for i in range(cells):
        left = s0 + (s1 - s0) * i / cells
        right = s0 + (s1 - s0) * (i + 1) / cells
        top = axis_weight(family, (left, right), family.intervals[1], family.intervals[3]).hi
        if not top:
            continue
        candidates = []
        for q, slack in options:
            g = gamma_lower(q, rate, t0, t1, left, right)
            if g > 0:
                # Upward rounding preserves domination while keeping coefficients small.
                weight = F(ceil((top / g) * 2**32), 2**32)
                candidates.append((weight * slack, q.id, weight))
        if not candidates:
            return None, {}
        _, name, weight = min(candidates)
        selected[name] = max(selected.get(name, F(0)), weight)
    # Each selected call pays its full budget once, including across disjoint cells.
    total = 2 * sum(selected.get(q.id, 0) * slack for q, slack in options)
    return total, {key: str(value) for key, value in sorted(selected.items())}


def bounds(quotes, family, amin, amax, variance, model_run, cells=16):
    if not 0 < amin <= amax or cells < 1:
        raise ValueError("Bornes de variance ou nombre de cellules invalides.")
    witness = check_witness(quotes, amin, amax, variance, model_run)
    d, band = denominator_bound(family, amax)
    lo, hi = amin, amax
    evidence = {"modele_compatible": witness, "denominateur_au_moins": str(d),
                "bande_de_probabilite": band, "cellules_spatiales": cells}
    if d > 0:
        for name, rate, at_cap in (("plafond", amax, True), ("plancher", amin, False)):
            budget, weights = budget_bound(quotes, family, rate, at_cap, cells)
            evidence[name] = {"budget_au_plus": None if budget is None else str(budget),
                              "poids_des_calls": weights}
            if budget is not None:
                if at_cap:
                    lo = max(lo, amax - budget / d)
                else:
                    hi = min(hi, amin + budget / d)
    if lo > hi or ("variance" in witness and not lo <= F(witness["variance"]) <= hi):
        raise ArithmeticError("Les bornes calculées contredisent les certificats.")
    return lo, hi, evidence


def tail_probabilities(quotes, left, right):
    """Upper bounds for the two tail probabilities from future put and call asks."""
    put_tail, call_tail = F(1), F(1)
    for q in quotes:
        if q.k > left:
            put_tail = min(put_tail, (q.ask + q.k - 1) / (q.k - left))
        if q.k < right:
            call_tail = min(call_tail, q.ask / (right - q.k))
    return put_tail, call_tail


def gamma_mass_floor(quotes, target, rate, corridor, steps=24, bands=48):
    """Bound E[int S² Gamma dt] by integrating nested price-band indicators."""
    k, maturity = target.k, target.t
    geometry = []
    for j in range(1, bands + 1):
        width = k * j / (2 * bands)
        left, right = k - width, k + width
        if corridor is not None:
            left, right = max(left, corridor[0]), min(right, corridor[1])
        abs_log = max(-log_point(left / k).lo, log_point(right / k).hi)
        geometry.append((left, right, sqrt(k * left).lo, abs_log**2))
    total, slabs, probability_cache = F(0), [], {}
    # Drop the final slab by positivity to avoid the integrable maturity singularity.
    for i in range(steps - 1):
        t0, t1 = maturity * i / steps, maturity * (i + 1) / steps
        eligible = tuple(q for q in quotes if q.t >= t1)
        if eligible not in probability_cache:
            probability_cache[eligible] = [
                I(max(F(0), 1 - sum(tail_probabilities(eligible, l, r)))).rounded(64).lo
                for l, r, _, _ in geometry
            ]
        probabilities = probability_cache[eligible]
        vmin, vmax = rate * (maturity - t1), rate * (maturity - t0)
        scale = sqrt(2 * pi() * vmax).hi
        levels = []
        for _, _, numerator, log_sq in geometry:
            energy = F(ceil(64 * (log_sq / (2 * vmin) + vmax / 8)), 64)
            height = F(0) if energy > 100 else (I(numerator * exp_point(-energy).lo / scale).rounded(64).lo)
            levels.append(min(height, levels[-1]) if levels else height)
        differences = (a - b for a, b in zip(levels, levels[1:] + [F(0)]))
        floor = I(sum(gap * mass for gap, mass in zip(differences, probabilities))).rounded(64).lo
        total += (t1 - t0) * floor
        slabs.append({"temps": [str(t0), str(t1)], "integrande_au_moins": str(floor)})
    return total, slabs


def gamma_tail_mass(quotes, target, rate, corridor, steps=24):
    """Upper bound for E[int S² Gamma 1_{S outside corridor} dt]."""
    k, maturity = target.k, target.t
    left, right = corridor
    log_distances = (log_point(k / left).lo, log_point(right / k).lo)
    total = F(0)
    for i in range(steps):
        t0, t1 = maturity * i / steps, maturity * (i + 1) / steps
        vmin, vmax = rate * (maturity - t1), rate * (maturity - t0)
        tails = tail_probabilities([q for q in quotes if q.t >= t1], left, right)
        # The global maximum of S² Gamma over S is k/sqrt(2*pi*v).
        global_mass = (2 * k * (sqrt(vmax).hi - sqrt(vmin).lo)
                       / (rate * sqrt(2 * pi()).lo))
        for side, distance, prob in zip((left, right), log_distances, tails):
            upper = global_mass
            # The spatial maximum on this tail is at its finite boundary
            # if the lognormal peak log(S/k)=v/2 lies outside that tail.
            if distance > 0 and (side == left or distance >= vmax / 2):
                at = min(vmax, distance**2)
                scale = sqrt(2 * pi() * at).lo
                if scale > 0:
                    # exp(-distance²/(2v))/sqrt(v) peaks at v=distance².
                    energy = min(F(100), F(floor(64 * distance**2 / (2 * at)), 64))
                    height = sqrt(side * k).hi * exp_point(-energy).hi / scale
                    upper = min(upper, (t1 - t0) * height)
            total += I(prob * upper).rounded(64).hi
    return total


def gamma_bounds(quotes, call_id, reference, amin, amax, variance=None, model_run=None, corridor=None):
    if not 0 < amin <= amax or reference <= 0:
        raise ValueError("La variance de référence et les bornes a priori doivent être positives et ordonnées.")
    target = next((q for q in quotes if q.id == call_id), None)
    if target is None:
        raise ValueError(f"Call introuvable dans les cotations pour {call_id}.")
    if corridor is not None and not 0 < corridor[0] < target.k < corridor[1]:
        raise ValueError("Le corridor doit être positif et contenir le strike dans son intérieur.")
    witness = check_witness(quotes, amin, amax, variance, model_run, suggested=reference)
    black = black_call(target.k, reference * target.t)
    d, slabs = gamma_mass_floor(quotes, target, reference, corridor)
    outside = gamma_tail_mass(quotes, target, reference, corridor) if corridor is not None else F(0)
    lo, hi = amin, amax
    if d > 0:
        low_gap = 2 * (target.bid - black.hi) - max(F(0), amax - reference) * outside
        high_gap = 2 * (target.ask - black.lo) + max(F(0), reference - amin) * outside
        lo = max(lo, reference + min(F(0), low_gap / d))
        hi = min(hi, reference + max(F(0), high_gap / d))
    if lo > hi or ("variance" in witness and not lo <= F(witness["variance"]) <= hi):
        raise ArithmeticError("La garantie gamma contredit le modèle compatible.")
    kernel = {"noyau": "gamma_Black", "call": target.id, "strike_normalise": str(target.k),
              "maturite": str(target.t), "variance_de_reference": str(reference)}
    if corridor is not None:
        kernel["corridor_prix_normalise"] = list(map(str, corridor))
    evidence = {"modele_compatible": witness, "denominateur_au_moins": str(d),
                "bid_ask": [str(target.bid), str(target.ask)], "prix_Black": [str(black.lo), str(black.hi)],
                "masse_gamma_hors_corridor_au_plus": str(outside),
                "subdivisions_de_temps": 24, "bandes_de_prix": 48, "minorants_par_tranche": slabs}
    return kernel, lo, hi, evidence
