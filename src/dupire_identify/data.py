"""Bounded Yahoo collection, explicit conventions, and frozen-curve scenarios."""
from datetime import datetime, timezone, timedelta
from fractions import Fraction as F
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import csv
import json
import multiprocessing as mp
from pathlib import Path
import re
import time
import numpy as np
from scipy.optimize import linprog
from .io import Quote,save_json,write_quotes,rational
from .progress import QUIET

CONTRACT=re.compile(r"^([A-Z0-9.^]+?)(\d{6})([CP])(\d{8})$")


def _yahoo_task(symbol,expiry):
    import yfinance as yf
    t=yf.Ticker(symbol)
    if expiry is None:return list(t.options)
    chain=t.option_chain(expiry)
    records=[]
    for kind,frame in (("call",chain.calls),("put",chain.puts)):
        for row in frame.to_dict("records"):
            records.append(dict(contract=str(row["contractSymbol"]),kind=kind,
                strike=float(row["strike"]),bid=float(row["bid"]),ask=float(row["ask"]),
                volume=float(row.get("volume",0) or 0),
                open_interest=float(row.get("openInterest",0) or 0),
                last_trade=str(row.get("lastTradeDate",""))))
    return records


def _bounded_request(symbol,expiry,timeout,progress):
    # A separate process can be terminated if Yahoo's option-chain call hangs.
    pool=ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context("spawn"))
    future=pool.submit(_yahoo_task,symbol,expiry)
    deadline=time.monotonic()+timeout
    try:
        while True:
            try: return future.result(timeout=min(progress.heartbeat,max(.01,deadline-time.monotonic())))
            except TimeoutError:
                progress.event("yahoo_request","working",symbol=symbol,expiry=expiry,timeout_s=timeout)
                if time.monotonic()>=deadline:raise TimeoutError(f"Yahoo timeout {timeout}s for {symbol}/{expiry}")
    finally:
        # Public terminate_workers exists on newer Python; 3.11/3.12 need an
        # explicit owned-process cleanup before nonblocking executor shutdown.
        processes=list(getattr(pool,"_processes",{}).values())
        for process in processes:
            if process.is_alive(): process.terminate()
        pool.shutdown(wait=False,cancel_futures=True)
        for process in processes:process.join(timeout=.5)


def collect_yahoo(output,symbol="^SPX",root="SPXW",min_days=14,max_days=365,
                  max_expiries=12,timeout=35,retries=2,allow_closed=False,progress=QUIET):
    if (symbol,root) not in (("^SPX","SPXW"),("^XSP","XSP")):
        raise ValueError("P1 collector supports ^SPX/SPXW PM or ^XSP/XSP PM only; no automatic ETF substitution")
    if not 0<min_days<max_days or max_expiries<1 or timeout<=0 or not 0<=retries<=5:
        raise ValueError("Invalid collection limits")
    try:
        import exchange_calendars as xcals
        import pandas as pd
    except ImportError as e:raise RuntimeError("Install .[yahoo] for Yahoo/calendar collection") from e
    now=datetime.now(timezone.utc)
    cal=xcals.get_calendar("XNYS",start=(now-timedelta(days=10)).date().isoformat(),
                           end=(now+timedelta(days=max_days+30)).date().isoformat())
    stamp=pd.Timestamp(now).floor("min")
    opened=bool(cal.is_open_on_minute(stamp))
    if not opened and not allow_closed:
        nextopen=cal.next_open(stamp)
        raise RuntimeError(f"Regular session closed; next_open_utc={nextopen.isoformat()}. "
                           "--allow-closed records a stale/unverified snapshot explicitly.")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=dict(symbol=symbol,root=root,style="european",settlement="PM",
                  collection_started_utc=now.isoformat(),regular_session_open=opened,
                  quote_timestamps="Yahoo does not expose authoritative bid/ask timestamps",
                  synchronized_snapshot_proved=False,errors=[], rejected=[], expiries=[])
    save_json(output/"snapshot.json",manifest)
    def request(expiry):
        error=None
        for attempt in range(retries+1):
            try:return _bounded_request(symbol,expiry,timeout,progress)
            except Exception as e:
                error=e;progress.event("yahoo_retry","failed",attempt=attempt+1,expiry=expiry,error=str(e))
                if attempt<retries:
                    # Bounded backoff, with heartbeat rather than a silent sleep.
                    delay=2**attempt
                    with progress.phase("yahoo_backoff",seconds=delay):time.sleep(delay)
        raise RuntimeError(str(error))
    expiries=request(None)
    selected=[]
    for expiry in expiries:
        session=pd.Timestamp(expiry)
        if not cal.is_session(session):
            manifest["rejected"].append(dict(expiry=expiry,reason="expiry_not_a_trading_session"));continue
        close=cal.session_close(session).to_pydatetime()
        days=(close-now).total_seconds()/86400
        if min_days<=days<=max_days:selected.append((expiry,close))
    if len(selected)>max_expiries:
        indices=np.unique(np.linspace(0,len(selected)-1,max_expiries).round().astype(int))
        selected=[selected[i] for i in indices]
    rows=[]
    for i,(expiry,close) in enumerate(selected):
        try:
            records=request(expiry)
            accepted=0
            for r in records:
                match=CONTRACT.match(r["contract"])
                if not match or match.group(1)!=root:
                    manifest["rejected"].append(dict(contract=r["contract"],reason="root_or_settlement_mismatch"));continue
                if match.group(2)!=expiry[2:].replace("-",""):
                    manifest["rejected"].append(dict(contract=r["contract"],reason="expiry_mismatch"));continue
                if not np.isfinite(r["bid"]) or not np.isfinite(r["ask"]) or r["bid"]<0 or r["ask"]<=0 or r["bid"]>r["ask"]:
                    manifest["rejected"].append(dict(contract=r["contract"],reason="invalid_bid_ask"));continue
                r.update(symbol=symbol,root=root,style="european",settlement="PM",expiry=expiry,
                         expiry_utc=close.isoformat(),snapshot_utc=now.isoformat(),
                         t=repr((close-now).total_seconds()/(365*86400)),
                         captured_utc=datetime.now(timezone.utc).isoformat())
                rows.append(r);accepted+=1
            manifest["expiries"].append(dict(expiry=expiry,accepted=accepted))
        except Exception as e:
            manifest["errors"].append(dict(expiry=expiry,error=str(e)))
        save_json(output/"snapshot.json",manifest)
        progress.event("yahoo_expiries","running",done=i+1,total=len(selected),contracts=len(rows))
    if rows:
        with (output/"raw.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest.update(collection_finished_utc=datetime.now(timezone.utc).isoformat(),contracts=len(rows),
                    complete=bool(rows) and not manifest["errors"])
    save_json(output/"snapshot.json",manifest)
    if not rows:
        raise RuntimeError("No usable European PM chain returned. Yahoo availability is not guaranteed; "
                           "see snapshot.json. No prices fabricated or American ETF fallback used.")
    return manifest


def normalize(raw_file,curves_file,output,progress=QUIET):
    with open(raw_file,encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
    with open(curves_file,encoding="utf-8-sig",newline="") as f:curves=list(csv.DictReader(f))
    lookup={}
    for c in curves:
        key=(c["expiry"],c["root"],c["settlement"])
        if key in lookup:raise ValueError("Duplicate curve convention")
        lookup[key]=(rational(c["forward"]),rational(c["discount"]))
    quotes=[];metadata=[]
    conventions={(r["symbol"],r["root"],r["settlement"],r["snapshot_utc"]) for r in rows}
    if len(conventions)!=1:raise ValueError("Normalize a single symbol/root/settlement/snapshot at a time")
    for r in rows:
        if r["kind"]!="call":continue
        if r["style"].lower()!="european":raise ValueError("No American-to-European conversion inside normalization")
        key=(r["expiry"],r["root"],r["settlement"])
        if key not in lookup:raise ValueError(f"Missing explicit F,D curves: {key}")
        forward,discount=lookup[key]
        if forward<=0 or discount<=0:raise ValueError("F,D must be positive")
        q=Quote(r["contract"],rational(r["t"]),rational(r["strike"])/forward,
                rational(r["bid"])/(discount*forward),rational(r["ask"])/(discount*forward))
        quotes.append(q)
        metadata.append(dict(id=q.id,forward=str(forward),discount=str(discount),expiry=r["expiry"]))
    if not quotes:raise ValueError("No European calls after normalization")
    write_quotes(output,quotes)
    save_json(Path(output).with_suffix(".normalization.json"),
              dict(scope="conditional_on_fixed_forward_discount_scenario",conventions=list(conventions)[0],
                   exact_arithmetic=True,rows=metadata,curve_uncertainty_propagated=False))
    progress.event("normalization","done",done=len(quotes),total=len(quotes))
    return len(quotes)


def estimate_curves(raw_file,output,dmin=.8,dmax=1.2,min_pairs=5,progress=QUIET):
    """A frozen feasible put-call-parity scenario; NOT an uncertainty certificate."""
    with open(raw_file,encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
    groups={}
    for r in rows:
        if r["style"]!="european":raise ValueError("Parity scenario requires European calls AND puts")
        key=(r["expiry"],r["root"],r["settlement"],r["symbol"],r["snapshot_utc"])
        groups.setdefault(key,{}).setdefault(r["strike"],{})[r["kind"]]=r
    result=[]
    for key,chain in groups.items():
        pairs=[v for v in chain.values() if "call" in v and "put" in v]
        if len(pairs)<min_pairs:raise ValueError(f"{key}: insufficient_matched_strikes ({len(pairs)})")
        scale=float(np.median([float(p['call']['strike']) for p in pairs]))
        A=[];b=[]
        for p in pairs:
            c,put=p['call'],p['put'];k=float(c['strike'])/scale
            lower=(float(c['bid'])-float(put['ask']))/scale
            upper=(float(c['ask'])-float(put['bid']))/scale
            A.extend([[1,-k,1],[-1,k,1]]);b.extend([upper,-lower])
        with progress.phase("parity_scenario",expiry=key[0],pairs=len(pairs)):
            res=linprog([0,0,-1],A_ub=A,b_ub=b,bounds=[(.05,5),(dmin,dmax),(0,None)],method="highs")
        if not res.success:raise ValueError(f"{key}: no feasible common European parity scenario")
        Alevel,D,margin=res.x
        result.append(dict(expiry=key[0],root=key[1],settlement=key[2],forward=repr(scale*Alevel/D),discount=repr(D)))
    with open(output,"w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["expiry","root","settlement","forward","discount"])
        writer.writeheader();writer.writerows(result)
    save_json(Path(output).with_suffix(".scenario.json"),dict(scope="numerical_frozen_parity_scenario",
        discount_prior=[dmin,dmax],uncertainty_propagated=False,curves=result))
    return result
