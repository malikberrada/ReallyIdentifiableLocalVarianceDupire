"""Prepare hash-checked TXO quotes and TAIEX closes for the published experiments."""
import argparse
import csv
from datetime import date
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dupire_identify.taifex_books import (
    ALIASES, close_from_twse, fetch, index_url, option_url, read_day,
)
from dupire_identify.taifex_optimization import trading_dates

ARCHIVE = ROOT / "data" / "taifex_txo"
PROVENANCE = ROOT / "data" / "provenance.json"
DEFAULT_INPUTS = ROOT / "data" / "downloaded" / "taifex_txo"


def project_quotes(raw, day):
    """Retain every regular-session TXO row, including missing and crossed quotes."""
    rows = read_day(raw, day)
    fields = tuple(ALIASES)
    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow([ALIASES[key][0] for key in fields])
    writer.writerows(sorted(tuple(row[key] for key in fields) for row in rows))
    return stream.getvalue().encode("utf-8")


def project_closes(raw, month, start, end):
    """Keep the two published TWSE fields inside the study window in session order."""
    report = json.loads(raw.decode("utf-8-sig"))
    date_col = report["fields"].index("日期")
    close_col = report["fields"].index("收盤指數")
    closes = {row[date_col]: row[close_col] for row in report["data"]}
    values = []
    for day in trading_dates(raw, month):
        if start <= day <= end:
            close_from_twse(raw, day)  # Validates the unique positive close.
            label = f"{day.year - 1911}/{day.month:02d}/{day.day:02d}"
            values.append([label, closes[label].replace(",", "")])
    result = {"stat": "OK", "fields": ["日期", "收盤指數"], "data": values}
    return (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def selected_files(market):
    day = date.fromisoformat(market["observation"])
    return {"options.csv": f"{day.isoformat()}.csv", "index.json": f"{day:%Y%m}.json"}


def checked(data, record, name):
    if len(data) != record["bytes"] or sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"Unexpected length or SHA-256 for {name}.")
    return data


def source_data(name, record, start, end):
    if name.endswith(".csv"):
        day = date.fromisoformat(name[:-4])
        if record["url"] != option_url(day):
            raise ValueError(f"Unexpected TAIFEX source for {name}.")
        return project_quotes(fetch(record["url"]), day)
    month = date(int(name[:4]), int(name[4:6]), 1)
    if record["url"] != index_url(month):
        raise ValueError(f"Unexpected TWSE source for {name}.")
    return project_closes(fetch(record["url"]), month, start, end)


def prepare(folder, *, offline=False, history=False):
    folder = Path(folder)
    manifest = json.loads((ARCHIVE / "manifest.json").read_bytes())
    records = manifest["files"]
    start, end = map(date.fromisoformat, manifest["date_range"])
    market = json.loads(PROVENANCE.read_bytes())["market"]
    files = {name: name for name in records} if history else selected_files(market)
    folder.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder.parent) as temporary:
        staging = Path(temporary) / "inputs"
        staging.mkdir()
        for dest, name in files.items():
            record = records[name]
            data = (ARCHIVE / name).read_bytes() if offline else source_data(name, record, start, end)
            (staging / dest).write_bytes(checked(data, record, name))
        if not folder.exists():
            try:
                staging.rename(folder)
                return folder
            except OSError:
                if not folder.is_dir():
                    raise
        # A changed existing observation must never be overwritten by a retry.
        for dest, name in files.items():
            path = folder / dest
            if path.exists():
                checked(path.read_bytes(), records[name], path.name)
        for dest, name in files.items():
            path = folder / dest
            if not path.exists():
                try:
                    os.link(staging / dest, path)
                except FileExistsError:
                    checked(path.read_bytes(), records[name], path.name)
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Destination for the verified inputs")
    parser.add_argument("--offline", action="store_true", help="Read the records included in this checkout")
    parser.add_argument("--history", action="store_true", help="Prepare the entire 33-session scan cache")
    args = parser.parse_args()
    default = DEFAULT_INPUTS.parent / "txo-history" if args.history else DEFAULT_INPUTS
    try:
        result = prepare(args.out or default, offline=args.offline, history=args.history)
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"Input acquisition failed: {error}")
    print(f"Verified {'the 33-session history' if args.history else 'the October 7 TXO inputs'} in {result.resolve()}")
