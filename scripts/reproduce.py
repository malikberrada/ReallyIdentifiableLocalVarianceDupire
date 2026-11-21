"""Reproduce Table 1 from the official TXO reports and the analytical control panels."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from dupire_identify.taifex_txo import fingerprint, load_json, run, verify
from download_data import ARCHIVE, selected_files
from synthetic_inputs import generate


def synthetic_results(out, provenance):
    directory = out / "synthetic-inputs"
    generate(directory)
    for key in ("synthetic_quotes", "informative_quotes"):
        record = provenance[key]
        path = directory / record["filename"]
        data = path.read_bytes()
        if len(data) != record["bytes"] or sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Regenerated synthetic input differs: {path.name}")

    results = {}
    for contract, digest in provenance["synthetic_certificates"].items():
        target = out / f"{contract}.json"
        subprocess.run([
            sys.executable, "-I", "-X", "utf8", str(ROOT / "scripts" / "garantie_positive.py"),
            "gamma", "--quotes", str(directory / "r3-quotes.csv"), "--call-id", contract,
            "--variance-reference", "0.04", "--corridor", "0.65", "1.50",
            "--epsilon", "0.01", "--decimales", "6", "--out", str(target),
        ], check=True, stdout=subprocess.DEVNULL)
        data = target.read_bytes()
        if sha256(data).hexdigest() != digest:
            raise ValueError(f"Regenerated synthetic certificate differs: {contract}")
        results[contract] = {**json.loads(data)["garantie_uniforme"]["decimal"], "sha256": digest}
    return results


def reproduce(inputs, output):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = load_json(ROOT / "data" / "provenance.json")
    published = provenance["market"]
    case = load_json(ROOT / published["case"])
    certificate = load_json(ROOT / published["certificate"])
    if case["panel_sha256"] != published["panel_sha256"] or fingerprint(certificate) != published["result_sha256"]:
        raise ValueError("Published TAIFEX case or certificate differs from its provenance.")
    expected = load_json(ROOT / published["table1"])
    if inputs is None:
        files = selected_files(published)
        quotes = ARCHIVE / files["options.csv"]
        index = ARCHIVE / files["index.json"]
    else:
        quotes, index = inputs / "options.csv", inputs / "index.json"
    with TemporaryDirectory(dir=output.parent, prefix=".table1-") as temporary:
        stage = Path(temporary) / "results"
        market = stage / "txo"
        acquired = run(market, case["settings"], quotes, index, case)
        checked = verify(market)
        if acquired != checked or load_json(market / "result.json") != certificate:
            raise ValueError("The recomputed TXO witness or bounds differ from the published certificate.")
        summary = {"txo": checked, "synthetic_gamma": synthetic_results(stage, provenance)}
        if summary != expected:
            raise ValueError("Table 1 differs from the published summary.")
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                           encoding="utf-8", newline="\n")
        stage.rename(output)
    print(json.dumps(summary, indent=2))
    print(output / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, help="Read options.csv and index.json from this directory (default: bundled study inputs)")
    parser.add_argument("--out", type=Path, help="New result directory (default: timestamp under runs/)")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("review-%Y%m%dT%H%M%S-%fZ")
    try:
        reproduce(args.inputs, args.out or ROOT / "runs" / stamp)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"Reproduction failed: {error}")
