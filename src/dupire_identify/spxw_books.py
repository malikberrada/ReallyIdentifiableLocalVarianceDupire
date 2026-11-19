"""Immutable Yahoo books, OCC checks and exact put-call-parity geometry."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from fractions import Fraction as F
import json
import re
from zoneinfo import ZoneInfo

from .interval import I

NY = ZoneInfo('America/New_York')
OCC = re.compile(r'([A-Z0-9]+) *(\d{6})([CP])(\d{8})', re.ASCII)
SPEC = 'https://www.cboe.com/tradable_products/sp_500/spx_weekly_options/specifications/'


def decode(raw):
    def unique(entries):
        out = dict(entries)
        if len(out) != len(entries):
            raise ValueError('Duplicate JSON key')
        return out
    def invalid(value):
        raise ValueError(f'Invalid JSON number: {value}')
    return json.loads(raw, parse_float=Decimal, parse_constant=invalid, object_pairs_hook=unique)


def utc(value):
    stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError(f'Missing time zone: {value}')
    return stamp.astimezone(timezone.utc)


def settlement(expiry, supplied):
    session = date.fromisoformat(expiry)
    if session.isoformat() != expiry or session.weekday() >= 5:
        raise ValueError(f'Invalid SPXW expiry session: {expiry}')
    if supplied is not None:
        close = utc(supplied)
    else:
        import exchange_calendars as xc
        calendar = xc.get_calendar('XNYS', start=f'{session.year-1}-01-01', end=f'{session.year+1}-12-31')
        if not calendar.is_session(expiry):
            raise ValueError(f'XNYS holiday: {expiry}')
        close = calendar.session_close(expiry).to_pydatetime()
    local = close.astimezone(NY)
    if local.date() != session or (local.hour, local.minute, local.second, local.microsecond) not in ((13,0,0,0),(16,0,0,0)):
        raise ValueError(f'Invalid US equity closing time for {expiry}: {close}')
    return close


@dataclass(frozen=True)
class Book:
    id: str
    strike: F
    bid: F
    ask: F


@dataclass(frozen=True)
class Pair:
    k: F                         # listed strike, in index points
    call: Book
    put: Book

    @property
    def strip(self):
        return self.call.bid-self.put.ask, self.call.ask-self.put.bid


def parse_book(row, moment, expiry, kind):
    name = str(row.get('contractSymbol', ''))
    parsed = OCC.fullmatch(name)
    if not parsed or parsed[1] != 'SPXW':
        raise ValueError('root_or_contract_format')
    if parsed[2] != date.fromisoformat(expiry).strftime('%y%m%d') or parsed[3] != kind:
        raise ValueError('expiry_or_option_type_mismatch')
    if row.get('currency') != 'USD' or row.get('contractSize') != 'REGULAR':
        raise ValueError('currency_or_size')
    if str(row.get('exerciseStyle', 'european')).lower() != 'european' or str(row.get('settlement', 'PM')).upper() != 'PM':
        raise ValueError('contradictory_contract_specification')
    try:
        k, bid, ask = (F(str(row[x])) for x in ('strike', 'bid', 'ask'))
    except (KeyError, ValueError, ZeroDivisionError):
        raise ValueError('missing_or_invalid_book') from None
    if k <= 0 or k != F(int(parsed[4]), 1000):
        raise ValueError('occ_strike_mismatch')
    if bid < 0 or ask < 0:
        raise ValueError('negative_book')
    if bid == ask == 0:
        raise ValueError('empty_book')
    if bid == 0:
        raise ValueError('zero_bid')
    if bid > ask:
        raise ValueError('crossed_book')
    if row.get('lastTradeDate') is not None:
        try:
            last = utc(row['lastTradeDate'])
        except (ValueError, TypeError):
            raise ValueError('invalid_last_trade_time') from None
        if last > moment:
            raise ValueError('future_last_trade')
    return Book(name, k, bid, ask)


def expiry_book(snapshot, settings, calendar):
    if snapshot.get('source') != 'yfinance' or not snapshot.get('chains'):
        raise ValueError('Expected a yfinance raw snapshot containing chains')
    begin, end = (utc(snapshot[key]) for key in ('started_utc', 'ended_utc'))
    if end < begin:
        raise ValueError('Reversed snapshot timestamps')
    dates = sorted({c['expiry'] for c in snapshot['chains'] if c.get('symbol') == '^SPX'})
    if not dates:
        raise ValueError('SPXW requires the Yahoo ^SPX index chain')
    expiry = settings['expiry'] or dates[0]
    selected = [c for c in snapshot['chains'] if c['expiry'] == expiry and c.get('symbol') == '^SPX']
    if len(selected) != 1:
        raise ValueError(f'Missing or duplicated ^SPX slice: {expiry}')
    c = selected[0]
    start, stop = (utc(c[key]) for key in ('retrieval_started_utc','retrieval_ended_utc'))
    if not begin <= start <= stop <= end:
        raise ValueError('Inconsistent chain retrieval chronology')
    if any(isinstance(e,dict) and e.get('expiry') == expiry for e in snapshot.get('errors', [])):
        raise ValueError('The selected expiry has a recorded collection error')
    under = c['underlying']
    if under.get('symbol') != '^SPX' or under.get('quoteType') != 'INDEX' or under.get('currency') != 'USD':
        raise ValueError('Missing or inconsistent ^SPX index metadata')
    spot = F(str(under['regularMarketPrice']))
    if spot <= 0:
        raise ValueError('Invalid spot for strike selection')
    market_time = under.get('regularMarketTime')
    if market_time is not None:
        market_time = datetime.fromtimestamp(int(market_time),timezone.utc)
        if market_time > stop:
            raise ValueError('Underlying market timestamp after retrieval')
    if calendar is not None:
        supplied = calendar['sessions'][expiry]
        source = str(calendar['source'])
        if 'session_close_utc' in c and utc(c['session_close_utc']) != utc(supplied):
            raise ValueError('Conflicting archived closing times')
    else:
        supplied = c.get('session_close_utc')
        source = c.get('calendar_source','XNYS session close from exchange_calendars')
    close = settlement(expiry, supplied)
    dt = close - begin
    t = F((dt.days*86400+dt.seconds)*10**6+dt.microseconds,365*86400*10**6)
    if t <= 0:
        raise ValueError('Expired contract at the snapshot valuation time')
    audit = dict(symbol='^SPX', root='SPXW', style='european', settlement='PM cash', spec=SPEC,
                 expiry=expiry, available_expiries=dates, snapshot_start=begin.isoformat(), snapshot_end=end.isoformat(),
                 valuation_utc=begin.isoformat(), retrieval=[start.isoformat(),stop.isoformat()],
                 underlying_market_utc=market_time.isoformat() if market_time else None,
                 underlying_delay_minutes=str(under['exchangeDataDelayedBy']) if under.get('exchangeDataDelayedBy') is not None else None,
                 regular_market_state=under.get('marketState'),
                 regular_session_open_at_start=snapshot.get('regular_session_open_at_start'),
                 spot_for_selection_only=str(spot), session_close_utc=close.isoformat(), calendar_source=source,
                 act365f=str(t), bid_ask_synchronization_verified=False, rejected=[], retained=[],
                 collection_errors=snapshot.get('errors',[]), unselected_slices=[
                     dict(symbol=g.get('symbol'),expiry=g.get('expiry'),calls=len(g.get('calls',[])),puts=len(g.get('puts',[])))
                     for g in snapshot['chains'] if g is not c])
    return c, spot, t, audit


def select_pairs(c, spot, settings, audit):
    records = [(kind,row) for kind,key in [('C','calls'),('P','puts')] for row in c[key]]
    repeated = Counter(str(row.get('contractSymbol','')) for _,row in records)
    groups = defaultdict(lambda: defaultdict(list))
    rejected = audit['rejected']
    def reject(name, reason):
        rejected.append({'id':name,'reason':reason})
    stop = utc(c['retrieval_ended_utc'])
    for kind,row in records:
        name = str(row.get('contractSymbol',''))
        try:
            if repeated[name] > 1:
                raise ValueError('duplicate_contract')
            q = parse_book(row,stop,c['expiry'],kind)
            groups[q.strike][kind].append(q)
        except ValueError as error:
            reject(name,str(error))
    pairs = {}
    for k,legs in sorted(groups.items()):
        if len(legs['C']) == len(legs['P']) == 1:
            pairs[k] = Pair(k,legs['C'][0],legs['P'][0])
        else:
            for leg in legs.values():
                for q in leg:
                    reject(q.id,'missing_or_ambiguous_pair')
    audit['valid_matched_pairs'] = len(pairs)
    mode = settings['selection']
    if mode == 'all':
        keys = sorted(pairs)
    elif mode == 'strikes':
        keys = list(map(F,settings['levels']))
        if not set(keys) <= pairs.keys():
            raise ValueError('An explicitly requested listed strike has no valid matched pair')
    elif mode == 'anchors' and pairs:
        keys = [min(pairs,key=lambda k:(abs(k-spot*F(level)),k)) for level in settings['levels']]
    else:
        raise ValueError('No eligible matched SPXW call/put pairs or invalid selection')
    if len(set(keys)) != len(keys) or len(keys) < 2:
        raise ValueError('At least two distinct matched strikes are required; the selection overlaps or is too small')
    for k,pair in pairs.items():
        if k not in keys:
            for q in (pair.call,pair.put):
                reject(q.id,'outside_requested_strikes')
    result = [pairs[k] for k in sorted(keys)]
    audit.update(raw_rows=len(records),retained=[q.id for p in result for q in (p.call,p.put)],
                 rejection_counts=dict(Counter(q['reason'] for q in rejected)))
    ids=set(audit['retained'])
    audit['retained_last_trade_utc']={r['contractSymbol']:(utc(r['lastTradeDate']).isoformat() if r.get('lastTradeDate') is not None else None)
                                    for _,r in records if r.get('contractSymbol') in ids}
    audit['last_trade_is_bid_ask_time']=False
    return result


def clip(poly, inequality):
    """Exact half-plane intersection, including points and line segments."""
    out = []
    for i, point in enumerate(poly):
        other = poly[i-1]
        u, v = inequality(other), inequality(point)
        if (u < 0 < v) or (v < 0 < u):
            f = u/(u-v)
            out.append(tuple(a+f*(b-a) for a,b in zip(other,point)))
        if v >= 0:
            out.append(point)
    return list(dict.fromkeys(out))


def parity(pairs, discount=None):
    """Vertices (D,A=DF) enclosing all feasible curves, no assumed rate band."""
    p, q = pairs[0], pairs[-1]
    width = q.k-p.k
    dl = max(max(v.put.bid/v.k for v in pairs),(p.strip[0]-q.strip[1])/width)
    dh = (p.strip[1]-q.strip[0])/width
    if discount:
        dl, dh = max(dl, F(discount[0])), min(dh,F(discount[1]))
    al, ah = max(v.call.bid for v in pairs), p.strip[1]+p.k*dh
    if dl > dh or al > ah:
        return []
    poly = list(dict.fromkeys([(dl,al),(dh,al),(dh,ah),(dl,ah)]))
    for pair in pairs:
        lo, hi = pair.strip
        poly = clip(poly,lambda x:x[1]-pair.k*x[0]-lo)
        poly = clip(poly,lambda x:hi-x[1]+pair.k*x[0])
        if not poly:
            break
    return poly


def span(poly, func):
    values = [func(d,a) for d,a in poly]
    return I(min(values),max(values))


def endpoints(interval):
    return [str(interval.lo),str(interval.hi)]


def parity_conflict(pairs, discount=None):
    """Two necessary discount inequalities proving an empty polygon exactly."""
    def row(value,rule,*books):
        return dict(value=str(value),rule=rule,strikes=[str(p.k) for p in books],
                    ids=[[p.call.id,p.put.id] for p in books])
    lower=row(F(0),'positive_discount')
    upper=None
    if discount:
        lower=row(F(discount[0]),'discount_prior_min')
        upper=row(F(discount[1]),'discount_prior_max')
    largest=max(pairs,key=lambda p:p.call.bid)
    for j,p in enumerate(pairs):
        for value,rule,books in [(p.put.bid/p.k,'put_bid_over_strike',(p,)),
                                 ((largest.call.bid-p.strip[1])/p.k,'call_bid_and_parity_upper',(largest,p))]:
            if value>F(lower['value']):
                lower=row(value,rule,*books)
        for q in pairs[j+1:]:
            diff=q.k-p.k
            low=(p.strip[0]-q.strip[1])/diff
            high=(p.strip[1]-q.strip[0])/diff
            if low>F(lower['value']):
                lower=row(low,'two_strips_lower',p,q)
            if upper is None or high<F(upper['value']):
                upper=row(high,'two_strips_upper',p,q)
    if upper is None or F(lower['value'])<=F(upper['value']):
        raise ArithmeticError('No exact discount contradiction found')
    return dict(discount_must_be_at_least=lower,discount_must_be_at_most=upper,
                contradiction=str(F(lower['value'])-F(upper['value'])))
