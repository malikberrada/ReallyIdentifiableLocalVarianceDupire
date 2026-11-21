import csv
from datetime import date
from fractions import Fraction as F
from io import BytesIO, StringIO
import json
from pathlib import Path
import ssl
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dupire_identify import taifex_books as books
from dupire_identify.taifex_books import ALIASES, choose_series, clocks, prepare, read_day
from dupire_identify.taifex_txo import calculate, check_settings, file_digests, run, select, verify


DAY = date(2026, 11, 19)
EXPIRY = date(2026, 12, 2)
BASE = dict(date=DAY.isoformat(), expiry=EXPIRY.isoformat(), series="202612W1",
            all_pairs=False, strikes=None, amin="1/400", amax="1/4", epsilon="0.01")
INDEX = json.dumps({"stat": "OK", "fields": ["日期", "收盤指數"],
                    "data": [["115/11/19", "100.00"]]}).encode()


def leg(strike, side, bid="4.5", ask="5", **changes):
    row = dict(day="2026/11/19", root="TXO", series="202612W1", expiry="2026/12/02",
               strike=str(strike), side=side, bid=bid, ask=ask, session="Regular", halt="-")
    row.update(changes)
    return row


def make_csv(rows, *, chinese=False, with_expiry=True):
    out = StringIO(newline="")
    keys = [k for k in ALIASES if with_expiry or k != "expiry"]
    writer = csv.writer(out)
    writer.writerow([ALIASES[k][-1 if chinese else 0] for k in keys] + ["Volume"])
    for row in rows:
        if chinese:
            row = dict(row, side={"Call": "買權", "Put": "賣權"}[row["side"]],
                       session={"Regular": "一般", "After-Hours": "盤後"}[row["session"]])
        writer.writerow([row[k] for k in keys] + ["0"])
    return out.getvalue().encode("cp950" if chinese else "utf-8-sig")


def panel(strikes):
    return [leg(k, side) for k in strikes for side in ("Call", "Put")]


class DownloadTests(unittest.TestCase):
    def test_windows_uses_system_curl_when_python_rejects_the_issuer(self):
        url = books.option_url(DAY)
        error = URLError(ssl.SSLCertVerificationError("unable to get local issuer certificate"))
        with (patch.object(books, "urlopen", side_effect=error),
              patch.object(sys, "platform", "win32"),
              patch.dict(books.os.environ, {"SystemRoot": "Windows"}),
              patch.object(books.subprocess, "Popen") as popen):
            process = popen.return_value.__enter__.return_value
            process.stdout = BytesIO(b"official CSV bytes")
            process.wait.return_value = 0
            self.assertEqual(books.fetch(url), b"official CSV bytes")
        command = popen.call_args.args[0]
        self.assertEqual(command[:2], [str(Path("Windows") / "System32" / "curl.exe"), "--disable"])
        self.assertEqual(command[command.index("--proto-redir") + 1], "=https")
        self.assertEqual(command[-2:], ["--url", url])
        self.assertFalse({"-k", "--insecure", "--ssl-no-revoke"}.intersection(command))

    def test_native_validation_failure_names_the_host_and_discards_the_body(self):
        url = books.index_url(DAY)
        error = ssl.SSLCertVerificationError("issuer verification failed")
        with (patch.object(books, "urlopen", side_effect=error),
              patch.object(sys, "platform", "win32"),
              patch.dict(books.os.environ, {"SystemRoot": "Windows"}),
              patch.object(books.subprocess, "Popen") as popen):
            process = popen.return_value.__enter__.return_value
            process.stdout = BytesIO(b"partial response")
            process.wait.return_value = process.returncode = 60
            with self.assertRaisesRegex(URLError, r"www\.twse\.com\.tw.*issuer.*curl\.exe exit 60"):
                books.fetch(url)

    def test_retry_is_specific_to_windows_certificate_errors(self):
        cases = [("linux", ssl.SSLCertVerificationError("unknown issuer")),
                 ("win32", ConnectionRefusedError("connection refused"))]
        for system, error in cases:
            with (self.subTest(system=system), patch.object(sys, "platform", system),
                  patch.object(books, "urlopen", side_effect=URLError(error)),
                  patch.object(books, "_windows_download") as native):
                with self.assertRaises(URLError):
                    books.fetch(books.option_url(DAY))
                native.assert_not_called()

    def test_download_size_limit(self):
        with patch.object(books, "DOWNLOAD_LIMIT", 4):
            self.assertEqual(books._read_download(BytesIO(b"1234")), b"1234")
            with self.assertRaisesRegex(ValueError, "file limit"):
                books._read_download(BytesIO(b"12345"))


class TxoTests(unittest.TestCase):
    def test_filters_quotes_and_sessions_and_preserves_unscaled_points(self):
        rows = panel([99, 100]) + [
            leg(100, "Call", session="After-Hours"), leg(101, "Put", root="TX1"),
            leg(102, "Call", bid="0"), leg(103, "Put", ask="-"),
            leg(104, "Call", bid="9", ask="8"), leg(105, "Put", halt="Y"),
            leg(106, "Call"), leg(106, "Call"), leg(106, "Put"),
            leg(107, "Call", day="2026/11/18"),
        ]
        pairs, _, _, audit = prepare(make_csv(rows), INDEX, BASE)
        self.assertEqual([p.k for p in pairs], [99, 100])
        self.assertEqual(pairs[0].call.bid, F("4.5"))
        self.assertEqual(audit["rejection_counts"], {
            "nonpositive_bid_or_ask": 1, "missing_bid_or_ask": 1, "crossed_market": 1,
            "trading_halt": 1, "duplicate_contract": 2, "unpaired_valid_leg": 1})
        self.assertEqual(audit["regular_rows_at_expiry"], 11)

    def test_public_chinese_feed_and_stable_projection(self):
        rows = panel([99, 100])
        english = prepare(make_csv(rows), INDEX, BASE)[3]
        chinese = prepare(make_csv(list(reversed(rows)), chinese=True, with_expiry=False), INDEX, BASE)[3]
        self.assertEqual(chinese["panel_sha256"], english["panel_sha256"])
        self.assertEqual(chinese["expiry_basis"], "explicit_series_expiry")
        rows[0]["bid"] = "4.25"
        self.assertNotEqual(prepare(make_csv(rows), INDEX, BASE)[3]["panel_sha256"], english["panel_sha256"])

    def test_uses_only_requested_day_and_explicit_undated_series(self):
        raw = make_csv(panel([99]), with_expiry=False)
        with self.assertRaisesRegex(ValueError, "both --series and --expiry"):
            prepare(raw, INDEX, dict(BASE, expiry=None))
        with self.assertRaisesRegex(ValueError, "future maturity"):
            prepare(make_csv(panel([99])), INDEX, dict(BASE, expiry="2026-12-03"))
        with self.assertRaisesRegex(ValueError, "No TXO regular-session"):
            read_day(raw, date(2026, 11, 20))
        with self.assertRaisesRegex(ValueError, "no unique positive"):
            prepare(raw, INDEX.replace(b"115/11/19", b"115/11/18"), BASE)
        with self.assertRaisesRegex(ValueError, "CSV header"):
            read_day(b"<html>Website busy</html>", DAY)

    def test_expiry_selection_and_local_clocks(self):
        series = [leg(99, "Put", series="A", expiry="2026/12/02"),
                  leg(99, "Put", series="B", expiry="2026/12/04")]
        self.assertEqual(choose_series(series, DAY, None, None)[:2], ("A", EXPIRY))
        with self.assertRaisesRegex(ValueError, "Multiple series"):
            choose_series(series + [leg(98, "Put", series="C")], DAY, None, None)
        start, end, t = clocks(DAY, EXPIRY)
        self.assertEqual(start.isoformat(), "2026-11-19T13:45:00+08:00")
        self.assertEqual(end.isoformat(), "2026-12-02T13:30:00+08:00")
        self.assertEqual(t, F(13 * 1440 - 15, 365 * 1440))

    def test_selection_keeps_distinct_strikes_and_breaks_ties_downward(self):
        pairs, spot, *_ = prepare(make_csv(panel([97, 98, 99, 100, 101, 102, 103])), INDEX, BASE)
        self.assertEqual([p.k for p in select(pairs, spot, BASE)[0]], [98, 99, 100, 101, 102])
        self.assertEqual(select(pairs[:2], spot, BASE)[0], [])
        self.assertEqual(select(pairs[:2], spot, dict(BASE, all_pairs=True))[0], pairs[:2])
        self.assertEqual(select(pairs, spot, dict(BASE, strikes=["103", "98"]))[0], [pairs[1], pairs[-1]])
        self.assertEqual(select(pairs, spot, dict(BASE, strikes=["98", "110"]))[0], [])
        with self.assertRaisesRegex(ValueError, "distinct"):
            check_settings(dict(BASE, strikes=["99", "99.0"]))
        ties, _, *_ = prepare(make_csv(panel([97.5, 98.5, 99, 100, 101, 102])), INDEX, BASE)
        self.assertEqual(select(ties, spot, BASE)[0][0].k, F("97.5"))

    def test_incompatible_quotes_give_exact_parity_conflict(self):
        rows = [leg(99, "Call", "4", "4"), leg(99, "Put", "2", "2"),
                leg(100, "Call", "6", "6"), leg(100, "Put", "2", "2")]
        _, result = calculate(make_csv(rows), INDEX, dict(BASE, all_pairs=True))
        self.assertEqual(result["status"], "infeasible_parity")
        self.assertGreater(F(result["selected_parity"]["proof"]["contradiction"]), 0)
        self.assertFalse(result["witness_verified"])

    def test_offline_replay_and_rehashed_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            quotes, index = path / "input.csv", path / "input.json"
            quotes.write_bytes(make_csv(panel([99, 100])))
            index.write_bytes(INDEX)
            result = run(path / "first", BASE, quotes, index)
            self.assertEqual(result["status"], "insufficient_quotes")
            portable = json.loads((path / "first/reference.json").read_bytes())
            copy = path / "again"
            self.assertEqual(run(copy, BASE, quotes, index, portable), result)
            self.assertEqual(verify(copy), result)
            with self.assertRaises(FileExistsError):
                run(copy, BASE, quotes, index)
            (copy / "summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Manifest mismatch"):
                verify(copy)
            (copy / "manifest.json").write_text(json.dumps(file_digests(copy)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "recomputed"):
                verify(copy)
            quotes.write_bytes(make_csv([leg(99, "Put", "2.5", "4")]))
            with self.assertRaisesRegex(ValueError, "differs from the reference"):
                run(path / "changed", BASE, quotes, index, portable)
            self.assertFalse((path / "changed").exists())


if __name__ == "__main__":
    unittest.main()
