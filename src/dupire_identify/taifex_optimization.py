"""Search the listed TXO series on a fixed date window and verify calibrated counterexamples."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import date
from fractions import Fraction as F
import json
import os
from pathlib import Path
import sys
import tempfile

from .io import save_json
from .spxw_books import parity
from .spxw_gamma import prove_gamma
from .spxw_witness import verify_model
from .taifex_books import close_from_twse, fetch, index_url, option_url, plain_date, prepare, read_day
from .taifex_txo import calculate, fingerprint, reference_for, select

COLUMNS = ("date","expiry","series","status","eligible_pairs","contracts","target_call",
           "fixed_estimate","fixed_error","auto_estimate","auto_error","objective","both","panel_sha256")
VERSION = "TAIFEX_dual_precision_1"


def case_name(day, series):
    if not series.isascii() or not series.isalnum():
        raise ValueError("Invalid official series identifier")
    return day.replace("-", "")+"_"+series+".json"


def result_row(s):
    row = {k:s[k] for k in ("date","expiry","series","status","eligible_pairs","contracts","panel_sha256")}
    row["target_call"] = s.get("target_call", "")
    g = s.get("gamma", {})
    for name, key in (("fixed","0.04"), ("auto","auto")):
        row[name+"_estimate"] = g.get(key,{}).get("estimate","")
        row[name+"_error"] = g.get(key,{}).get("absolute_error_at_most","")
    row["objective"] = max((row["fixed_error"],row["auto_error"]), key=F) if {"0.04","auto"} <= g.keys() else ""
    row["both"] = bool(row["objective"] and g["0.04"]["target_met"] and g["auto"]["target_met"])
    return row


def table(refs, sessions, directory):
    rows = sorted((result_row(r["summary"]) for r in refs),key=lambda s:(s["date"],s["expiry"],s["series"]))
    with (directory/"panels.csv").open("w",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    both=[r for r in rows if r["both"]]
    certified=[r for r in rows if r["objective"]]
    best=min(certified,key=lambda r:(F(r["objective"]),r["date"],r["expiry"],r["series"])) if certified else None
    stats = dict(protocol=VERSION, sessions=sessions, panels=len(rows),
                 two_verified_bounds=len(certified), both_within_target=len(both),
                 dates_with_both=len({r["date"] for r in both}), winner=best,
                 status_counts={s:sum(r["status"]==s for r in rows) for s in sorted({r["status"] for r in rows})})
    save_json(directory/"summary.json",stats)
    return stats


def months(start, end):
    y, m = start.year, start.month
    while (y,m) <= (end.year,end.month):
        yield date(y,m,1)
        y,m = (y+1,1) if m==12 else (y,m+1)


def cached(path, url, validate, offline=False):
    if path.exists():
        data=path.read_bytes()
        validate(data)
        return data
    if offline:
        raise FileNotFoundError(f"Missing offline study input: {path}")
    data=fetch(url)
    validate(data)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        tmp=Path(f.name)
        f.write(data)
    try:
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)
    return data


def trading_dates(raw, month):
    report=json.loads(raw.decode("utf-8-sig"))
    if report.get("stat")!="OK":
        raise ValueError(f"TWSE returned no index report for {month:%Y-%m}")
    pos=report["fields"].index("日期")
    found=[]
    for row in report["data"]:
        y,m,d=map(int,row[pos].split("/"))
        day=date(y+1911,m,d)
        if (day.year,day.month)!=(month.year,month.month):
            raise ValueError("TWSE returned an unexpected month")
        found.append(day)
    if len(set(found))!=len(found):
        raise ValueError("Duplicate TWSE trading date")
    return sorted(found)


def one_day(args):
    day, raw_path, index_path, case_dir, published = args
    raw, idx=raw_path.read_bytes(), index_path.read_bytes()
    rows=read_day(raw,day)
    contracts=set()
    for row in rows:
        if not row["expiry"] or row["expiry"] in ("-","--"):
            raise ValueError("Historical series scan needs the official Contract Date column")
        exp=plain_date(row["expiry"])
        if 7 <= (exp-day).days <= 60:
            contracts.add((row["series"],exp))
    refs=[]
    for series, expiry in sorted(contracts):
        filename=case_name(day.isoformat(),series)
        settings=dict(date=day.isoformat(),expiry=expiry.isoformat(),series=series,
                      amin="1/400",amax="1/4",epsilon="0.01",all_pairs=False,strikes=None)
        old=json.loads((published/filename).read_bytes()) if published else None
        if old is not None and old["settings"]!=settings:
            raise ValueError(f"Published configuration differs for {filename}")
        audit,result=calculate(raw,idx,settings,old["decisions"] if old else None)
        ref=reference_for(settings,audit,result)
        if old is not None and ref!=old:
            raise ValueError(f"Published reference differs for {filename}")
        save_json(case_dir/filename,ref)
        refs.append(ref)
    return refs


def sweep(start,end,out,cache,jobs,published,offline=False):
    if start>end or jobs<1:
        raise ValueError("Invalid date window or worker count")
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    out.parent.mkdir(parents=True,exist_ok=True)
    if offline:
        if not cache.is_dir():
            raise FileNotFoundError(f"Offline input directory does not exist: {cache}")
        manifest = cache / "manifest.json"
        if manifest.is_file():
            first, last = map(date.fromisoformat, json.loads(manifest.read_bytes())["date_range"])
            if start < first or end > last:
                raise ValueError(f"The bundled history covers {first} through {last}.")
    else:
        cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent,prefix=".txo-scan-") as staging:
        root=Path(staging)/"scan"
        cases=root/"panels"
        cases.mkdir(parents=True)
        tasks=[]
        for month in months(start,end):
            index_file=cache/(month.strftime("%Y%m")+".json")
            raw_index=cached(index_file,index_url(month),lambda b:trading_dates(b,month),offline)
            for day in trading_dates(raw_index,month):
                if not start<=day<=end:continue
                close_from_twse(raw_index,day)
                source=cache/(day.isoformat()+".csv")
                cached(source,option_url(day),lambda b:read_day(b,day),offline)
                tasks.append((day,source,index_file,cases,published))
        if not tasks:
            raise ValueError("No published TWSE trading sessions in this interval")
        if jobs==1:
            refs=[r for task in tasks for r in one_day(task)]
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                refs=[r for group in pool.map(one_day,tasks) for r in group]
        if published:
            expected={p.name for p in published.glob("*.json") if start.strftime("%Y%m%d")<=p.name[:8]<=end.strftime("%Y%m%d")}
            if expected!={p.name for p in cases.glob("*.json")}:
                raise ValueError("Date/expiry inventory differs from the supplied references")
        result=table(refs,len(tasks),root)
        save_json(root/"protocol.json",dict(start=start.isoformat(),end=end.isoformat(),
                   expiry_days=[7,60],anchors=["0.98","0.99","1","1.01","1.02"],
                   prior=["1/400","1/4"],corridor=["0.65","1.50"],epsilon="0.01",
                   objective="minimize the larger of the two displayed certified error bounds"))
        os.rename(root,out)
    return result


def extend(parent, raw, index, portable=None):
    config=parent["settings"]
    audit, result=calculate(raw,index,config,parent["decisions"])
    if reference_for(config,audit,result)!=parent or result["status"]!="certified":
        raise ValueError("Original two-kernel certificate does not verify on this source panel")
    all_pairs, spot, t, _=prepare(raw,index,config)
    previous, reason=select(all_pairs,spot,config)
    if reason:
        raise ValueError(reason)
    target=next(p for p in previous if p.call.id==result["target_call"])
    model=result["model"]
    lo, hi=F(config["amin"]), F(config["amax"])
    kept=[]
    for pair in all_pairs:
        try:
            verify_model(model,[pair],t,lo,hi)
        except ValueError as exc:
            if not str(exc).startswith("Witness outside raw bid/ask"):
                raise
        else:
            kept.append(pair)
    if not set(previous)<=set(kept):
        raise ValueError("Original constraints lost in the extended panel")
    witness=verify_model(model,kept,t,lo,hi)
    zone=parity(kept)
    fixed_auto=result["gamma"]["auto"]["reference_variance"]
    proofs, summary, positions={}, {}, {}
    for label,reference in (("fixed","0.04"),("original_auto",fixed_auto),("updated_auto","auto")):
        settings=dict(amin=config["amin"],amax=config["amax"],epsilon=config["epsilon"],
                      corridor=["0.65","1.50"],digits=6,reference=reference)
        previous_positions=portable["positions"][label] if portable is not None else None
        gamma=prove_gamma(zone,kept,target,t,settings,previous_positions)
        proofs[label]=gamma
        positions[label]=[cell["subhedge"] for cell in gamma["time_slabs"]]
        e=gamma["estimate"]
        summary[label]=dict(variance_reference=gamma["reference_variance"],estimate=e["estimate"],
                            error=e["absolute_error_at_most"],outer=e["display_outer"],
                            target_met=e["exact_target_met"] and e["displayed_target_met"])
    evidence=dict(protocol="TXO_witness_extension_1",parent_hash=parent["result_sha256"],
                  panel_sha256=audit["panel_sha256"],date=audit["date"],expiry=audit["expiry"],
                  target_call=target.call.id,original_pairs=len(previous),eligible_pairs=len(all_pairs),
                  strikes=[str(p.k) for p in kept],contracts=witness["checked_contracts"],
                  proof_sha256=fingerprint({"witness":witness,"proofs":proofs}),summary=summary,
                  positions=positions)
    if portable is not None and evidence!=portable:
        raise ValueError("Recomputed extension differs from its portable reference")
    return evidence


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("scan",help="Optimize the two certified error bounds over dates and listed series")
    p.add_argument("--start",required=True,type=date.fromisoformat)
    p.add_argument("--end",required=True,type=date.fromisoformat)
    p.add_argument("--jobs",type=int,default=1)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--cache",type=Path,help="Reusable directory for official CSV and TWSE monthly JSON files")
    p.add_argument("--offline",action="store_true",help="Use only existing files in --cache")
    p.add_argument("--published",type=Path,help="Verify these portable references for every panel")
    v=sub.add_parser("separation",help="Verify distinct attainable model averages against the original quotes")
    v.add_argument("--case",type=Path,required=True)
    v.add_argument("--out",type=Path,required=True)
    v.add_argument("--steps",type=int,default=2048)
    v.add_argument("--quotes",type=Path)
    v.add_argument("--index",type=Path)
    e=sub.add_parser("extend",help="Keep all additional quote pairs fitted by the original witness")
    e.add_argument("--case",type=Path,required=True,help="Original portable TXO reference")
    e.add_argument("--out",type=Path,required=True)
    e.add_argument("--published",type=Path,help="Verify published extension and stored hedge positions")
    e.add_argument("--quotes",type=Path)
    e.add_argument("--index",type=Path)
    args=parser.parse_args(argv)
    try:
        if args.command=="scan":
            cache=args.cache or args.out.parent/"txo_official_inputs"
            summary=sweep(args.start,args.end,args.out,cache,args.jobs,args.published,args.offline)
        else:
            if args.out.exists():
                raise FileExistsError(f"Choose a new output JSON: {args.out}")
            case=json.loads(args.case.read_bytes())
            day=date.fromisoformat(case["settings"]["date"])
            raw=args.quotes.read_bytes() if args.quotes else fetch(option_url(day))
            idx=args.index.read_bytes() if args.index else fetch(index_url(day))
            if args.command=="separation":
                from .taifex_separation import separate
                data=separate(case,raw,idx,args.steps)
                summary={k:data[k] for k in ("date","expiry","target_call","contracts","conclusions")}
            else:
                published=json.loads(args.published.read_bytes()) if args.published else None
                data=extend(case,raw,idx,published)
                summary={k:data[k] for k in ("date","expiry","target_call","original_pairs","contracts","summary")}
            args.out.parent.mkdir(parents=True,exist_ok=True)
            with args.out.open("x",encoding="utf-8") as f:
                json.dump(data,f,indent=2,allow_nan=False)
                f.write("\n")
        print(json.dumps(summary,indent=2))
        return 0
    except (OSError,ArithmeticError,ValueError,KeyError,TypeError,ImportError) as exc:
        print(f"TXO: {exc}",file=sys.stderr)
        return 1
