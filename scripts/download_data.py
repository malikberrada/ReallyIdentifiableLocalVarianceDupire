"""Retrieve the two frozen inputs to the 2026-10-26 SPXW experiment."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "data" / "provenance.json"
DEFAULT_INPUTS = ROOT / "examples" / "yahoo_spxw"
INPUTS = ("raw_snapshot.json", "session_closes.json")


def check_bytes(data, record, name):
    if len(data) != record["bytes"] or sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"Unexpected length or SHA-256 for {name}.")
    return data


def check_directory(folder, manifest):
    for name in INPUTS:
        check_bytes((folder / name).read_bytes(), manifest[name], name)


def prepare(folder, *, offline=False):
    folder = Path(folder)
    manifest = json.loads(PROVENANCE.read_bytes())
    folder.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder.parent) as temporary:
        staging = Path(temporary) / "inputs"
        staging.mkdir()
        for name in INPUTS:
            record = manifest[name]
            if "url" in record and not offline:
                with urlopen(record["url"], timeout=60) as response:
                    data = response.read(record["bytes"] + 1)
            else:
                data = (ROOT / record["bundled"]).read_bytes()
            (staging / name).write_bytes(check_bytes(data, record, name))
        # Authenticate both inputs before adding any missing file to the evidence directory.
        if not folder.exists():
            try:
                staging.rename(folder)
            except OSError:
                if not folder.is_dir():
                    raise
        if staging.exists():
            for name in INPUTS:
                path = folder / name
                if path.exists():
                    check_bytes(path.read_bytes(), manifest[name], name)
            for name in INPUTS:
                if not (folder / name).exists():
                    (staging / name).replace(folder / name)
        check_directory(folder, manifest)
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--offline", action="store_true", help="Reuse the previously downloaded raw file in the evidence directory.")
    args = parser.parse_args()
    try:
        result = prepare(args.out, offline=args.offline)
        source = "bundled" if args.offline else "downloaded"
        print(f"{source} and verified {result / INPUTS[0]}")
        print(f"SHA-256 {sha256((result / INPUTS[0]).read_bytes()).hexdigest()}")
    except (OSError, ValueError) as error:
        sys.exit(f"Input acquisition failed: {error}")
