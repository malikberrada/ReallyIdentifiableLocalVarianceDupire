"""Rebuild the SPXW certificates and the two synthetic gamma estimates on CPU."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys

# Resolve sibling scripts even when invoked using python -I.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from download_data import DEFAULT_INPUTS, PROVENANCE, check_bytes, check_directory

MARKET = ROOT / "examples" / "yahoo_spxw"
SPXW = ROOT / "scripts" / "yahoo_spxw.py"
GAMMA = ROOT / "scripts" / "garantie_positive.py"
EXPIRY = "2026-11-09"
TARGET = "SPXW261109C07680000"
EXPECTED = {
    "0.04": ("0.018988", "0.009501", ["0.009487", "0.028488"]),
    "auto": ("0.019929", "0.000662", ["0.019267", "0.020590"]),
}


def run(script, *arguments, expected=0):
    command = [sys.executable, "-I", "-X", "utf8", str(script), *map(str, arguments)]
    completed = subprocess.run(command, capture_output=True, encoding="utf-8")
    if completed.returncode != expected:
        raise RuntimeError(f"{script.name} exited {completed.returncode}\n"
                           f"{completed.stdout}\n{completed.stderr}")
    return completed.stdout


def verify_market(path, reference):
    result = json.loads(run(SPXW, "verify", "--run", path))
    estimate, error, outer = EXPECTED[reference]
    wanted = dict(verified=True, request_completed=True, contracts=10, expiry=EXPIRY,
                  call=TARGET, estimate=estimate, absolute_error_at_most=error,
                  outer=outer, market_synchronization_verified=False)
    if result != wanted:
        raise ValueError(f"SPXW ({reference}): published values differ: {result}")
    return result


def reproduce(inputs, output):
    manifest = json.loads(PROVENANCE.read_bytes())
    check_directory(inputs, manifest)
    # A new directory preserves previous evidence and failed runs for inspection.
    output.mkdir(parents=True, exist_ok=False)
    from synthetic_inputs import generate

    generate(output / "synthetic-inputs")
    quotes = output / "synthetic-inputs/r3-quotes.csv"
    check_bytes(quotes.read_bytes(), manifest["synthetic_quotes"], quotes.name)
    informative = output / "synthetic-inputs/informative-quotes.csv"
    check_bytes(informative.read_bytes(), manifest["informative_quotes"], informative.name)
    print("Checking the archived witness and certificate", flush=True)
    archived = output / "market-archived"
    archived.mkdir()
    for name in ("audit.json", "certificate.json", "model.json", "manifest.json"):
        shutil.copyfile(MARKET / name, archived / name)
    for name in ("raw_snapshot.json", "session_closes.json"):
        shutil.copyfile(inputs / name, archived / name)
    verify_market(archived, "0.04")
    common = ["certify", "--snapshot", inputs / "raw_snapshot.json",
              "--calendar", inputs / "session_closes.json", "--expiry", EXPIRY]
    summary = {"raw_snapshot_sha256": manifest["raw_snapshot.json"]["sha256"], "spxw": {}}
    for reference, name in [("0.04", "market-fixed"), ("auto", "market-auto")]:
        folder = output / name
        print(f"Fitting a new common witness and verifying {name}", flush=True)
        run(SPXW, *common, "--variance-reference", reference,
            "--corridor", "0.65", "1.50", "--epsilon", "0.01", "--decimales", "6",
            "--out", folder)
        summary["spxw"][reference] = verify_market(folder, reference)
    full = output / "market-all-pairs"
    run(SPXW, *common, "--all-pairs", "--out", full, expected=1)
    negative = json.loads(run(SPXW, "verify", "--run", full))
    audit = json.loads((full / "audit.json").read_bytes())
    proof = negative["proof"]
    conflict = [proof[k]["value"] for k in ("discount_must_be_at_least", "discount_must_be_at_most")]
    if not (negative["verified"] and negative["certified_infeasible"]
            and audit["valid_matched_pairs"] == 78 and conflict == ["53/50", "49/50"]):
        raise ValueError("The published all-pairs parity contradiction was not reproduced.")
    summary["spxw_all_pairs"] = {"pairs": 78, "discount_lower": conflict[0],
                                 "discount_upper": conflict[1], "verified": True}
    summary["synthetic_gamma"] = {}
    for contract in ("synthetic_1_4", "synthetic_2_4"):
        print(f"Regenerating {contract}", flush=True)
        target = output / (contract + ".json")
        run(GAMMA, "gamma", "--quotes", quotes, "--call-id", contract,
            "--variance-reference", "0.04", "--corridor", "0.65", "1.50",
            "--epsilon", "0.01", "--decimales", "6", "--out", target)
        data = target.read_bytes()
        if sha256(data).hexdigest() != manifest["synthetic_certificates"][contract]:
            raise ValueError(f"{contract}: regenerated certificate differs from its published digest.")
        summary["synthetic_gamma"][contract] = {
            **json.loads(data)["garantie_uniforme"]["decimal"], "sha256": sha256(data).hexdigest()}
    path = output / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(path.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--out", type=Path, help="New result directory (default: timestamp under runs/)")
    args = parser.parse_args()
    name = datetime.now(timezone.utc).strftime("review-%Y%m%dT%H%M%S-%fZ")
    try:
        reproduce(args.inputs, args.out or ROOT / "runs" / name)
    except (OSError, RuntimeError, ValueError) as error:
        sys.exit(f"Reproduction failed: {error}")
