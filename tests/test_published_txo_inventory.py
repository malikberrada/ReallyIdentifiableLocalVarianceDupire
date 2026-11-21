"""The 126 portable references, the inventory and the manuscript use the same ranking."""
from datetime import date
import csv
import json
from pathlib import Path

import pytest

from dupire_identify.taifex_optimization import sweep, table


def test_published_txo_inventory(tmp_path):
    directory = Path(__file__).resolve().parents[1] / "examples" / "txo_optimization"
    cases = [json.loads(path.read_bytes()) for path in sorted((directory / "panels").glob("*.json"))]
    dates = {case["summary"]["date"] for case in cases}
    stats = table(cases, len(dates), tmp_path)
    assert stats == json.loads((directory / "summary.json").read_bytes())
    with (tmp_path / "panels.csv").open(newline="") as actual, (directory / "panels.csv").open(newline="") as published:
        assert list(csv.DictReader(actual)) == list(csv.DictReader(published))
    assert (stats["sessions"], stats["panels"], stats["two_verified_bounds"]) == (33, 126, 89)
    assert (stats["both_within_target"], stats["dates_with_both"]) == (19, 7)
    assert (stats["winner"]["date"], stats["winner"]["series"]) == ("2026-10-07", "202610F3")


def test_offline_scan_does_not_fetch_missing_inputs(tmp_path, monkeypatch):
    def unexpected_fetch(url):
        pytest.fail(f"Offline scan requested {url}")

    monkeypatch.setattr("dupire_identify.taifex_optimization.fetch", unexpected_fetch)
    cache = tmp_path / "cache"
    cache.mkdir()
    output = tmp_path / "scan"
    day = date(2026, 10, 7)
    with pytest.raises(FileNotFoundError, match="Missing offline study input"):
        sweep(day, day, output, cache, 1, None, offline=True)
    assert not output.exists()
