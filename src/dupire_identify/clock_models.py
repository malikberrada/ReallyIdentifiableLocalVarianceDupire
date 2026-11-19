"""Quote-dependent deterministic clocks with exactly saturated time windows.

This is a sufficient inner family, never an outer approximation of arbitrary
smiles. Failure means only that this construction did not produce a model.
Binary Black inversions propose safe variance intervals; all final prices and
coefficients are recomputed with rational interval arithmetic.
"""
from fractions import Fraction as F
import math
from scipy.optimize import brentq
from .interval import I, black_call
from .io import read_quotes, write_quotes, save_json, quote_payload, sha256
from .progress import QUIET

KIND = 'analytic_piecewise_time_clock_v1'
FILES = ('normalized.csv', 'diffusion.json', 'pricing.json')


def is_clock(config):
    return all(len(set(map(F,p['variance']))) == 1 for p in config['phases'])


def price_proof(config, quotes):
    from . import plateau
    plateau.validate_config(config)
    if not is_clock(config):
        raise ValueError('Analytic clock requires spatially constant phases')
    prices = []
    for q in quotes:
        low, high = plateau.variance_envelope(config, q.t)
        if low != high or low <= 0:
            raise ValueError('Clock variance is not exact and positive')
        p = black_call(q.k, low)
        prices.append(dict(id=q.id, total_variance=str(low), price=p.json(),
                           within_bid_ask=bool(q.bid <= p.lo <= p.hi <= q.ask), eta='0'))
    return dict(kind=KIND, all_quotes_compatible=all(p['within_bid_ask'] for p in prices),
                prices=prices, pde_discretization_error='0',
                proof='Exact integrated variance; rational interval Black prices',
                coefficient_regularity='Piecewise constant in time, continuous diffusion paths')


def safe_variance_ranges(quotes, amin, amax):
    """Inner variance intervals, not claims about all admissible clocks."""
    from .calibrate_general import black
    ranges = {}
    unit = 2**44
    for t in sorted({q.t for q in quotes}):
        left, right = amin*t, amax*t
        for q in (z for z in quotes if z.t == t):
            def c(v): return float(black(float(q.k), v))
            l, h = float(amin*t), float(amax*t)
            if c(h) < float(q.bid) or c(l) > float(q.ask):
                return None
            if c(l) < float(q.bid):
                root = brentq(lambda v: c(v)-float(q.bid), l, h, xtol=1e-14)
                left = max(left, F(math.ceil(root*unit)+16, unit))
            if c(h) > float(q.ask):
                root = brentq(lambda v: c(v)-float(q.ask), l, h, xtol=1e-14)
                right = min(right, F(math.floor(root*unit)-16, unit))
        if left > right:
            return None
        # These checks are authoritative. Every variance in [left,right]
        # has an acceptable price by strict Black monotonicity in variance.
        for q in (z for z in quotes if z.t == t):
            if black_call(q.k, left).lo < q.bid or black_call(q.k, right).hi > q.ask:
                return None
        ranges[t] = (left, right)
    return ranges


def construct(quotes, family, amin, amax, endpoint, progress=QUIET, ranges=None):
    """Exact interval reachability, then backward selection of a clock.

    Each quote-maturity interval has a common free variance outside the
    requested time support and the prescribed extreme inside. Reachable
    integrated-variance intervals are propagated exactly (no LP rounding).
    """
    if endpoint not in ('minimum', 'maximum'):
        raise ValueError('Clock endpoint must be minimum or maximum')
    amin, amax = F(amin), F(amax)
    target = amin if endpoint == 'minimum' else amax
    if ranges is None:
        ranges = safe_variance_ranges(quotes, amin, amax)
    if ranges is None:
        return None
    ta, tb, ka, kb = family.support()
    dates = [F(0)] + sorted(ranges)
    reach = [(F(0), F(0))]
    increments = []
    for left, right in zip(dates, dates[1:]):
        pulse = max(F(0), min(right, tb)-max(left, ta))
        free = right-left-pulse
        low_inc, high_inc = target*pulse+amin*free, target*pulse+amax*free
        a = max(ranges[right][0], reach[-1][0]+low_inc)
        b = min(ranges[right][1], reach[-1][1]+high_inc)
        if a > b:
            return None
        reach.append((a, b)); increments.append((low_inc, high_inc, pulse, free))
    variance = [F(0)] * len(dates)
    variance[-1] = sum(reach[-1])/2
    for i in range(len(dates)-2, -1, -1):
        lo, hi, _, _ = increments[i]
        a, b = max(reach[i][0], variance[i+1]-hi), min(reach[i][1], variance[i+1]-lo)
        if a > b:
            raise ArithmeticError('Exact clock reachability inconsistency')
        variance[i] = (a+b)/2
    end = max(tb, dates[-1])
    if end == dates[-1]:
        end += F(1, 64)
    boundaries = sorted(set(dates + [ta, tb, end]))
    segments = []
    for left, right in zip(boundaries, boundaries[1:]):
        if ta <= left and right <= tb:
            rate = target
        elif left >= dates[-1]:
            rate = (amin+amax)/2
        else:
            j = next(j for j in range(len(dates)-1) if dates[j] <= left < dates[j+1])
            _, _, pulse, free = increments[j]
            if free <= 0:
                raise ArithmeticError('No free clock budget outside plateau')
            rate = (variance[j+1]-variance[j]-target*pulse)/free
        if not amin <= rate <= amax:
            raise ArithmeticError('Clock coefficient outside exact prior')
        segments.append((left, right, rate))
    # The first segment becomes the analytic lognormal prefix.
    l, prefix, initial = segments.pop(0)
    if l != 0 or prefix <= 0 or not segments:
        raise ValueError('Invalid analytic prefix geometry')
    knots = [min(F(1, 8), ka/2), max(F(8), kb*2)]
    cfg = dict(kind='piecewise_quintic_plateau_diffusion_v1', amin=str(amin), amax=str(amax),
        prefix_end=str(prefix), prefix_variance=str(initial), strike_knots=list(map(str, knots)),
        price_envelope_intersection=True,
        phases=[dict(left=str(a), right=str(b), variance=[str(v)]*2) for a, b, v in segments])
    proof = price_proof(cfg, quotes)
    if not proof['all_quotes_compatible']:
        raise ArithmeticError('Constructed clock failed independent exact price check')
    progress.event('target_clock', 'certified', endpoint=endpoint, phases=len(segments), target=str(target))
    return cfg


def run(config, quotes_file, directory):
    from pathlib import Path
    from . import __version__
    out = Path(directory); out.mkdir(parents=True, exist_ok=True)
    if (out/'model_manifest.json').exists():
        raise ValueError('Clock output already exists')
    quotes = read_quotes(quotes_file); proof = price_proof(config, quotes)
    if not proof['all_quotes_compatible']:
        raise ValueError('Clock is not certified compatible with all observations')
    write_quotes(out/'normalized.csv', quotes)
    save_json(out/'diffusion.json', config); save_json(out/'pricing.json', proof)
    save_json(out/'model_manifest.json', dict(version=__version__, status='completed',
        proof_kind=KIND, compatible=True, quotes=quote_payload(quotes),
        artifacts={n:sha256(out/n) for n in FILES}))
    return proof


def verify(directory):
    import json
    from pathlib import Path
    out = Path(directory); m = json.loads((out/'model_manifest.json').read_text(encoding='utf-8'))
    if m.get('proof_kind') != KIND or m['status'] != 'completed' or set(m['artifacts']) != set(FILES):
        raise ValueError('Invalid analytic clock manifest')
    for n, digest in m['artifacts'].items():
        if sha256(out/n) != digest:
            raise ValueError('Changed analytic clock artifact: '+n)
    quotes = read_quotes(out/'normalized.csv')
    proof = price_proof(json.loads((out/'diffusion.json').read_text(encoding='utf-8')), quotes)
    if quote_payload(quotes) != m['quotes'] or not m['compatible'] or not proof['all_quotes_compatible']:
        raise ValueError('Clock is not linked to compatible observations')
    if proof != json.loads((out/'pricing.json').read_text(encoding='utf-8')):
        raise ValueError('Incorrect analytic clock prices')
    return dict(verified=True, compatible=True, optimizer_restarted=False, pde_solver_restarted=False)
