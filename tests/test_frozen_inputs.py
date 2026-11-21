import csv
from fractions import Fraction
import importlib.util
import json
from pathlib import Path

import pytest

from dupire_identify.taifex_books import prepare as prepare_book
from dupire_identify.taifex_txo import select

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_data", ROOT / "scripts/download_data.py")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


def test_offline_inputs_preserve_existing_files(tmp_path, monkeypatch):
    def no_network(url):
        pytest.fail(f"Unexpected network access: {url}")

    monkeypatch.setattr(downloader, "fetch", no_network)
    dest = tmp_path / "inputs"
    downloader.prepare(dest, offline=True)
    downloader.prepare(dest, offline=True)
    index = dest / "index.json"
    index.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        downloader.prepare(dest, offline=True)
    assert index.read_bytes() == b"changed"


def test_bad_second_download_publishes_nothing(tmp_path, monkeypatch):
    valid_quotes = (downloader.ARCHIVE / "2026-10-07.csv").read_bytes()
    responses = iter([valid_quotes, b"truncated TWSE response"])
    monkeypatch.setattr(downloader, "fetch", lambda url: next(responses))
    dest = tmp_path / "inputs"
    with pytest.raises(ValueError):
        downloader.prepare(dest)
    assert not dest.exists()


def test_hashes_and_pair_table_match_bundled_history(tmp_path):
    cache = downloader.prepare(tmp_path / "history", history=True, offline=True)
    assert len(list(cache.glob("*.csv"))) == 33
    assert len(list(cache.glob("*.json"))) == 2
    ex = ROOT / "examples" / "txo_optimization"
    case = json.loads((ex / "panels" / "20261007_202610F3.json").read_bytes())
    pairs, spot, _, audit = prepare_book((cache / "2026-10-07.csv").read_bytes(),
                                         (cache / "202610.json").read_bytes(), case["settings"])
    chosen, failure = select(pairs, spot, case["settings"])
    assert failure is None
    assert audit["panel_sha256"] == case["panel_sha256"]
    with (ex / "20261007_202610F3_pairs.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["strike", "call_bid", "call_ask", "put_bid", "put_ask", "selected_for_witness"]
    assert [tuple(map(Fraction, row)) for row in rows[1:]] == [
        (p.k, p.call.bid, p.call.ask, p.put.bid, p.put.ask, int(p in chosen)) for p in pairs
    ]
    assert (len(pairs), len(chosen), audit["regular_rows_at_expiry"]) == (95, 5, 318)
