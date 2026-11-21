"""Acquisition and independent replay for European TAIEX options (TXO)."""
import argparse
from datetime import date
from fractions import Fraction as F
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile

from .io import save_json
from .spxw_books import endpoints, parity, parity_conflict, span
from .spxw_gamma import prove_gamma
from .spxw_witness import propose, verify_model
from .taifex_books import ANCHORS, OPEN_DATA, SPEC, fetch, index_url, option_url, prepare

VERSION = "dupire_identify.txo.v1"
FILES = ("options.csv", "index.json", "sources.json", "settings.json", "audit.json",
         "result.json", "summary.json", "reference.json")
REFS = ("0.04", "auto")
INCONCLUSIVE_GAMMA = {
    "The corridor must contain every possible normalized target strike",
    "Could not prove a strictly positive denominator for this corridor",
}
SCOPE = ("Gamma-weighted annual local variance of a normalized positive martingale diffusion "
         "whose terminal level prices the single TAIFEX settlement fixing; deterministic curves. "
         "TAIFEX fixes on the mean TAIEX over the final 30 minutes. Daily best bids and asks "
         "are treated as simultaneous constraints. Cash-index spot local variance is not identified.")


def fingerprint(obj):
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return sha256(payload.encode("ascii")).hexdigest()


def select(pairs, spot, config):
    if config["strikes"] is not None:
        strikes = [F(k) for k in config["strikes"]]
        available = {p.k: p for p in pairs}
        if any(k not in available for k in strikes):
            return [], "A requested strike has no valid call/put pair."
        return [available[k] for k in sorted(strikes)], None
    if len(pairs) < 2:
        return [], "Fewer than two distinct valid matched strikes."
    if config["all_pairs"]:
        return pairs, None
    chosen = [min(pairs, key=lambda p: (abs(p.k - spot * ratio), p.k)) for ratio in ANCHORS]
    if len({p.k for p in chosen}) != len(ANCHORS):
        return [], "The five predetermined anchors do not select five distinct valid strikes."
    return sorted(chosen, key=lambda p: p.k), None


def geometry(pairs, polygon):
    if len(pairs) < 2:
        return {"status": "insufficient_pairs", "pairs": len(pairs)}
    if not polygon:
        return {"status": "empty", "pairs": len(pairs), "proof": parity_conflict(pairs)}
    return {"status": "nonempty", "pairs": len(pairs),
            "discount": endpoints(span(polygon, lambda d, a: d)),
            "forward": endpoints(span(polygon, lambda d, a: a/d)),
            "vertices_D_DF": [[str(d), str(a)] for d, a in polygon]}


def check_settings(c):
    date.fromisoformat(c["date"])
    if c["expiry"]:
        date.fromisoformat(c["expiry"])
    if not 0 < F(c["amin"]) <= F(c["amax"]) or F(c["epsilon"]) < 0:
        raise ValueError("Invalid annual variance bounds or requested error.")
    if c["strikes"] is not None:
        keys = list(map(F, c["strikes"]))
        if c["all_pairs"] or len(set(keys)) != len(keys) or len(keys) < 2 or min(keys) <= 0:
            raise ValueError("Supply at least two distinct positive strikes.")


def calculate(raw, index, config, decisions=None):
    check_settings(config)
    pairs, spot, t, audit = prepare(raw, index, config)
    all_poly = parity(pairs) if len(pairs) >= 2 else []
    selected, failure = select(pairs, spot, config)
    audit["selected_strikes"] = [str(p.k) for p in selected]
    audit["retained"] = [book.id for p in selected for book in (p.call, p.put)]
    result = dict(schema=VERSION, scope=SCOPE, market_synchronization_verified=False,
                  optimal_extrema_certified=False, witness_verified=False,
                  all_pairs_parity=geometry(pairs, all_poly))
    if failure:
        result.update(status="insufficient_quotes", reason=failure)
        return audit, result
    poly = all_poly if selected == pairs else parity(selected)
    result["selected_parity"] = geometry(selected, poly)
    if not poly:
        result.update(status="infeasible_parity")
        return audit, result
    amin, amax = F(config["amin"]), F(config["amax"])
    if decisions is None:
        try:
            model = propose(selected, t, amin, amax, poly)
        except ValueError as exc:
            if not str(exc).startswith("No common witness verified"):
                raise
            result.update(status="no_verified_witness", reason=str(exc))
            return audit, result
    else:
        model = decisions["model"]
    witness = verify_model(model, selected, t, amin, amax)
    target = min(selected, key=lambda p: (abs(p.k - spot), p.k))
    result.update(status="certified", target_call=target.call.id, witness_verified=True,
                  model=model, witness=witness, gamma={})
    for ref in REFS:
        settings = dict(amin=config["amin"], amax=config["amax"], epsilon=config["epsilon"],
                        corridor=["0.65", "1.50"], digits=6, reference=ref)
        positions = None if decisions is None else decisions["positions"].get(ref)
        try:
            result["gamma"][ref] = prove_gamma(poly, selected, target, t, settings, positions)
        except ValueError as exc:
            if positions is not None or str(exc) not in INCONCLUSIVE_GAMMA:
                raise
            result.setdefault("gamma_unavailable", {})[ref] = str(exc)
    if "gamma_unavailable" in result:
        result["status"] = "partial_certificate" if result["gamma"] else "witness_only"
    return audit, result


def brief(audit, result, config):
    out = {k: audit[k] for k in ("date", "expiry", "series", "regular_rows_at_expiry",
                                 "eligible_pairs", "selected_strikes", "panel_sha256")}
    out.update(status=result["status"], prior=[config["amin"], config["amax"]],
               contracts=len(audit["retained"]), all_pairs_parity=result["all_pairs_parity"]["status"],
               witness_verified=result["witness_verified"], market_synchronization_verified=False)
    if not result["witness_verified"]:
        out["reason"] = result.get("reason", "The selected put-call parity polygon is empty.")
        return out
    out["target_call"] = result["target_call"]
    out["gamma"] = {}
    if "gamma_unavailable" in result:
        out["gamma_unavailable"] = result["gamma_unavailable"]
    for ref, g in result["gamma"].items():
        e = g["estimate"]
        out["gamma"][ref] = dict(reference_variance=g["reference_variance"],
                                 estimate=e["estimate"], absolute_error_at_most=e["absolute_error_at_most"],
                                 outer=e["display_outer"], epsilon=config["epsilon"],
                                 target_met=e["exact_target_met"] and e["displayed_target_met"])
    return out


def decisions_from(result):
    if not result["witness_verified"]:
        return None
    return {"model": result["model"], "positions": {
        ref: [slab["subhedge"] for slab in proof["time_slabs"]] for ref, proof in result["gamma"].items()}}


def reference_for(config, audit, result):
    return dict(schema=VERSION, settings=config, panel_sha256=audit["panel_sha256"],
                result_sha256=fingerprint(result), summary=brief(audit, result, config),
                decisions=decisions_from(result))


def file_digests(root):
    return {name: sha256((root/name).read_bytes()).hexdigest() for name in FILES}


def load_json(path):
    return json.loads(path.read_bytes())


def run(output, config, quotes=None, index_file=None, reference=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    check_settings(config)
    if reference is not None and reference.get("schema") != VERSION:
        raise ValueError("Unexpected portable reference schema.")
    day = date.fromisoformat(config["date"])
    opt_url, spot_url = option_url(day), index_url(day)
    raw = quotes.read_bytes() if quotes else fetch(opt_url)
    index = index_file.read_bytes() if index_file else fetch(spot_url)
    decisions = reference["decisions"] if reference else None
    if reference is not None:
        # Check the complete canonical slice and the index close before using a saved hedge.
        panel = prepare(raw, index, config)[3]["panel_sha256"]
        if panel != reference["panel_sha256"]:
            raise ValueError("The official date/series, book, halt flags or closing index differs from the reference.")
    audit, result = calculate(raw, index, config, decisions)
    portable = reference_for(config, audit, result)
    if reference is not None and portable != reference:
        raise ValueError("Independent replay differs from the published reference.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".txo-") as staging:
        tmp = Path(staging)/"run"
        tmp.mkdir()
        (tmp/"options.csv").write_bytes(raw)
        (tmp/"index.json").write_bytes(index)
        sources = dict(options={"input": "local_file" if quotes else "official_download",
                                "historical_url": opt_url, "product_specification": SPEC,
                                "daily_open_data_catalog": OPEN_DATA},
                       index={"input": "local_file" if index_file else "official_download",
                              "monthly_url": spot_url})
        for name, content in [("settings", config), ("sources", sources), ("audit", audit),
                              ("result", result), ("summary", portable["summary"]), ("reference", portable)]:
            save_json(tmp/(name+".json"), content)
        save_json(tmp/"manifest.json", file_digests(tmp))
        os.rename(tmp, output)
    return portable["summary"]


def verify(root):
    root = Path(root)
    if load_json(root/"manifest.json") != file_digests(root):
        raise ValueError("Manifest mismatch. One or more archived files have changed.")
    old = load_json(root/"result.json")
    config = load_json(root/"settings.json")
    raw, index = (root/"options.csv").read_bytes(), (root/"index.json").read_bytes()
    audit, result = calculate(raw, index, config, decisions_from(old))
    portable = reference_for(config, audit, result)
    if (audit != load_json(root/"audit.json") or result != old or
            portable != load_json(root/"reference.json") or portable["summary"] != load_json(root/"summary.json")):
        raise ValueError("The recomputed book, rational certificate or published summary differs.")
    return portable["summary"]


def inputs(group):
    group.add_argument("--quotes", type=Path, help="Read a local official TAIFEX CSV")
    group.add_argument("--index", type=Path, help="Read a local TWSE MI_5MINS_HIST JSON")
    group.add_argument("--out", type=Path, required=True, help="New output directory")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    r = subs.add_parser("run", help="Fetch or import the two official inputs and compute both references")
    r.add_argument("--date", required=True, help="Observation day in Taipei (YYYY-MM-DD)")
    r.add_argument("--expiry", help="Settlement date (default: listed maturity nearest 14 calendar days)")
    r.add_argument("--series", help="TAIFEX series such as 202612 or 202612W1")
    r.add_argument("--amin", default="1/400", help="Minimum annual local variance")
    r.add_argument("--amax", default="1/4", help="Maximum annual local variance")
    r.add_argument("--epsilon", default="0.01")
    sel = r.add_mutually_exclusive_group()
    sel.add_argument("--all-pairs", action="store_true")
    sel.add_argument("--strikes", nargs="+", help="Use these exact strikes instead of five spot anchors")
    inputs(r)
    p = subs.add_parser("replay", help="Reacquire source data and verify a published portable reference")
    p.add_argument("--case", type=Path, required=True, help="Reference JSON distributed with this patch")
    inputs(p)
    v = subs.add_parser("verify", help="Check the saved input, witness and bounds offline")
    v.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            summary = verify(args.run)
        elif args.command == "replay":
            ref = load_json(args.case)
            summary = run(args.out, ref["settings"], args.quotes, args.index, ref)
        else:
            keys = ("date", "series", "expiry", "amin", "amax", "epsilon", "all_pairs", "strikes")
            summary = run(args.out, {k: getattr(args, k) for k in keys}, args.quotes, args.index)
        print(json.dumps(summary, indent=2))
        return 0
    except (OSError, ArithmeticError, ValueError, KeyError, TypeError, IndexError, ImportError) as exc:
        print(f"TXO: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
