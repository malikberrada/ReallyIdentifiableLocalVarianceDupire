"""Bounded capture of the yfinance API output before any product/quote filter."""
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import version
import json
from math import isfinite
from multiprocessing import get_context
from pathlib import Path
import time

from .io import save_json
from .spxw_books import NY


def _now():
    return datetime.now(timezone.utc)


def _json_value(value):
    if value is None or isinstance(value,(str,int,bool)):
        return value
    if isinstance(value,float):
        return value if isfinite(value) else None
    if isinstance(value,dict):
        return {k:_json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_json_value(v) for v in value]
    if type(value).__name__ in ('NAType','NaTType'):
        return None
    if isinstance(value,(date,datetime)):
        return value.isoformat()
    if hasattr(value,'item'):
        return _json_value(value.item())
    raise TypeError(f'Unexpected Yahoo scalar: {type(value).__name__}')


def _worker(expiry,connection):
    try:
        import yfinance as yf
        ticker=yf.Ticker('^SPX')
        if expiry is None:
            payload=dict(expiries=list(ticker.options),version=version('yfinance'))
        else:
            c=ticker.option_chain(expiry)
            if c.calls is None or c.puts is None or not c.underlying:
                raise ValueError('Empty Yahoo option-chain response')
            payload=_json_value(dict(underlying=c.underlying,calls=c.calls.to_dict('records'),puts=c.puts.to_dict('records')))
        connection.send((True,payload))
    except Exception as exc:
        connection.send((False,f'{type(exc).__name__}: {exc}'))
    finally:
        connection.close()


def _one_request(expiry,timeout):
    ctx=get_context('spawn')
    receiver,sender=ctx.Pipe(duplex=False)
    proc=ctx.Process(target=_worker,args=(expiry,sender))
    try:
        proc.start()
        sender.close()
        if not receiver.poll(timeout):
            raise TimeoutError(f'Yahoo request exceeded {timeout} seconds')
        ok,payload=receiver.recv()
        if not ok:
            raise RuntimeError(payload)
        return payload
    finally:
        sender.close()
        receiver.close()
        if proc.pid is not None:
            proc.join(.2)
            if proc.is_alive():
                proc.terminate()
                proc.join(2)
            if proc.is_alive():
                proc.kill()
                proc.join(2)
            proc.close()


def capture(out,min_days,max_days,max_expiries,timeout,retries,allow_closed,request=None,clock=None):
    if not (0<=min_days<=max_days and 1<=max_expiries and timeout>0 and 0<=retries<=5):
        raise ValueError('Invalid collection limits')
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    snapshot_file=out/'raw_snapshot.json'
    now=clock or _now
    request=request or _one_request
    start=now()
    raw=dict(schema='dupire_identify.yahoo_raw.v1',source='yfinance',query_symbol='^SPX',started_utc=start.isoformat(),
             quote_timestamps_available=False,selection=dict(min_days=min_days,max_days=max_days,max_expiries=max_expiries),
             warning='No per-side quote timestamp. lastTradeDate refers to the last transaction.',
             chains=[],errors=[],retry_log=[],complete=False,ended_utc=start.isoformat())
    def flush():
        raw['ended_utc']=now().isoformat()
        save_json(snapshot_file,raw)
    def fetch(expiry):
        for attempt in range(retries+1):
            try:
                return request(expiry,timeout)
            except (RuntimeError,OSError,EOFError,ValueError) as exc:
                raw['retry_log'].append(dict(expiry=expiry,attempt=attempt+1,error=str(exc),utc=now().isoformat()))
                flush()
                if attempt==retries:
                    raise
                time.sleep(2**attempt)
    flush()
    try:
        import exchange_calendars as xc
        import pandas as pd
        cal=xc.get_calendar('XNYS',start=(start-timedelta(days=10)).date().isoformat(),
                            end=(start+timedelta(days=max_days+10)).date().isoformat())
        raw['calendar_source']='exchange_calendars '+version('exchange_calendars')+' XNYS cash close'
        open_now=bool(cal.is_open_on_minute(pd.Timestamp(start).floor('min')))
        raw['regular_session_open_at_start']=open_now
        if not open_now and not allow_closed:
            raise ValueError('US regular session closed; use --allow-closed for a tagged off-session collection')
        listed=fetch(None)
        raw['yfinance_version']=listed['version']
        raw['listed_expiries']=sorted(set(listed['expiries']))
        # Civil days in New York. The certificate later uses the exact UTC close and ACT/365F.
        day=start.astimezone(NY).date()
        raw['expiry_decisions']=[]
        selected=[]
        for expiry in raw['listed_expiries']:
            difference=(date.fromisoformat(expiry)-day).days
            if not min_days<=difference<=max_days:
                reason='outside_requested_days'
            elif not cal.is_session(expiry):
                reason='not_a_cash_session'
            elif len(selected)>=max_expiries:
                reason='expiry_limit'
            elif cal.session_close(expiry).to_pydatetime()<=start:
                reason='expired'
            else:
                selected.append(expiry)
                reason='selected'
            raw['expiry_decisions'].append(dict(expiry=expiry,reason=reason))
        raw['selection']['selected_expiries']=selected
        flush()
        if not selected:
            raise ValueError('Yahoo returned no eligible expiration for ^SPX')
        for expiry in selected:
            since=now()
            try:
                response=fetch(expiry)
                raw['chains'].append(dict(symbol='^SPX',expiry=expiry,retrieval_started_utc=since.isoformat(),
                     retrieval_ended_utc=now().isoformat(),session_close_utc=cal.session_close(expiry).isoformat(),
                     calendar_source=raw['calendar_source'],**response))
            except (RuntimeError,OSError,EOFError,ValueError) as exc:
                raw['errors'].append(dict(expiry=expiry,error=str(exc)))
            flush()
        raw['complete']=bool(raw['chains']) and not raw['errors']
        if not raw['complete']:
            raise RuntimeError('Partial or empty collection; see raw_snapshot.json for each expiry')
    except Exception as exc:
        raw['capture_status']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        flush()
    return raw
