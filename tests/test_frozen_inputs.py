import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_data", ROOT / "scripts/download_data.py")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


def test_offline_inputs_preserve_existing_files(tmp_path):
    dest = tmp_path / "inputs"
    downloader.prepare(dest, offline=True)
    downloader.prepare(dest, offline=True)
    calendar = dest / "session_closes.json"
    calendar.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        downloader.prepare(dest, offline=True)
    assert calendar.read_bytes() == b"changed"


def test_bad_or_truncated_download_publishes_nothing(tmp_path, monkeypatch):
    from io import BytesIO

    monkeypatch.setattr(downloader, "urlopen", lambda *a, **kw: BytesIO(b"wrong snapshot"))
    dest = tmp_path / "inputs"
    with pytest.raises(ValueError, match="SHA-256"):
        downloader.prepare(dest)
    assert not dest.exists()
