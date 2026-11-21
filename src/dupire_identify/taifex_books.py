"""Official TAIFEX regular-session TXO books, in unscaled index points."""
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, time, timedelta, timezone
from fractions import Fraction as F
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
from tempfile import TemporaryFile
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .spxw_books import Book, Pair


TAIPEI = timezone(timedelta(hours=8))
ANCHORS = tuple(map(F, ("0.98", "0.99", "1", "1.01", "1.02")))
SPEC = "https://www.taifex.com.tw/enl/eng2/tXO"
OPEN_DATA = "https://data.gov.tw/en/datasets/11320"
DOWNLOAD_LIMIT = 32 * 1024 * 1024
DOWNLOAD_TIMEOUT = 45
USER_AGENT = "DupireIdentify-TXO/1.0"
ALIASES = {
    "day": ("Date", "交易日期", "日期"),
    "root": ("Contract", "契約"),
    "series": ("Contract Month(Week)", "ContractMonth(Week)", "到期月份(週別)"),
    "expiry": ("Contract Date", "ContractDate", "契約到期日"),
    "strike": ("Strike Price", "StrikePrice", "履約價"),
    "side": ("Call/Put", "CallPut", "買賣權"),
    "bid": ("Best Bid", "BestBid", "最後最佳買價"),
    "ask": ("Best Ask", "BestAsk", "最後最佳賣價"),
    "session": ("Trading Session", "TradingSession", "交易時段"),
    "halt": ("Trading Halt", "TradingHalt", "是否因訊息面暫停交易"),
}
REGULAR = {"regular", "一般", "一般交易時段"}
SIDES = {"call": "C", "c": "C", "買權": "C", "put": "P", "p": "P", "賣權": "P"}
EMPTY = {"", "-", "--", "—", "N/A", "NA"}
NO_HALT = {"", "-", "n", "no", "0", "否", "正常", "false"}


def option_url(day):
    stamp = day.strftime("%Y/%m/%d")
    params = dict(down_type="1", commodity_id="TXO", queryStartDate=stamp, queryEndDate=stamp)
    return "https://www.taifex.com.tw/enl/eng3/optDataDown?" + urlencode(params)


def index_url(day):
    return "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?" + urlencode({
        "date": day.strftime("%Y%m%d"), "response": "json",
    })


def _read_download(stream):
    data = stream.read(DOWNLOAD_LIMIT + 1)
    if len(data) > DOWNLOAD_LIMIT:
        raise ValueError("Provider response exceeds the one-session file limit.")
    return data


def _windows_download(url):
    system_root = os.environ.get("SystemRoot")
    if not system_root:
        raise OSError("Windows system directory is unavailable (SystemRoot).")
    curl = Path(system_root) / "System32" / "curl.exe"
    # --disable must be first so that a user's curl configuration cannot alter TLS verification.
    command = [
        str(curl), "--disable", "--fail", "--silent", "--show-error", "--location",
        "--proto", "=https", "--proto-redir", "=https", "--max-time", str(DOWNLOAD_TIMEOUT),
        "--max-filesize", str(DOWNLOAD_LIMIT), "--user-agent", USER_AGENT, "--url", url,
    ]
    with TemporaryFile() as errors, subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors,
    ) as process:
        try:
            data = _read_download(process.stdout)
        except BaseException:
            process.kill()
            raise
        if process.wait() != 0:
            errors.seek(0)
            detail = errors.read(2048).decode(errors="replace").strip()
            raise OSError(f"curl.exe exit {process.returncode}: {detail}")
    return data


def fetch(url):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
            return _read_download(response)
    except OSError as exc:
        reason = exc.reason if isinstance(exc, URLError) else exc
        if not isinstance(reason, ssl.SSLCertVerificationError):
            raise
        if sys.platform != "win32":
            raise URLError(f"{request.host}: {reason}") from exc
        try:
            return _windows_download(url)
        except OSError as native_error:
            raise URLError(f"{request.host}: Python TLS: {reason}. Windows: {native_error}") from native_error


def plain_date(value):
    return date.fromisoformat(value.strip().replace("/", "-"))


def close_from_twse(raw, day):
    report = json.loads(raw.decode("utf-8-sig"))
    if report.get("stat") != "OK":
        raise ValueError("TWSE did not return an index report for the requested month.")
    fields = report["fields"]
    di, ci = fields.index("日期"), fields.index("收盤指數")
    roc_day = f"{day.year - 1911:03d}/{day.month:02d}/{day.day:02d}"
    prices = [F(row[ci].replace(",", "")) for row in report["data"] if row[di] == roc_day]
    if len(prices) != 1 or prices[0] <= 0:
        raise ValueError(f"TWSE has no unique positive TAIEX close for {day}.")
    return prices[0]


def read_day(raw, day):
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp950")
    reader = csv.DictReader(StringIO(text, newline=""))
    fields = [k.strip() for k in (reader.fieldnames or [])]
    if len(set(fields)) != len(fields):
        raise ValueError("Duplicate TAIFEX column.")
    reader.fieldnames = fields
    keys = {name: next((k for k in choices if k in fields), None)
            for name, choices in ALIASES.items()}
    missing = [k for k, v in keys.items() if v is None and k not in ("expiry", "halt")]
    if missing:
        raise ValueError("Unexpected TAIFEX CSV header. Missing: " + ", ".join(missing))
    rows = []
    for source in reader:
        row = {name: (source.get(key) or "").strip() if key else "" for name, key in keys.items()}
        if row["root"].upper() != "TXO" or row["session"].lower() not in REGULAR:
            continue
        if plain_date(row["day"]) == day:
            rows.append(row)
    if not rows:
        raise ValueError(f"No TXO regular-session rows for {day}. Check the file date and exchange calendar.")
    return rows


def choose_series(rows, day, expiry, series):
    dates = defaultdict(set)
    for row in rows:
        published = dates[row["series"]]
        if row["expiry"] not in EMPTY:
            published.add(plain_date(row["expiry"]))
    if series and series not in dates:
        raise ValueError(f"TXO series {series!r} is absent on {day}.")
    candidates = []
    for name, published in dates.items():
        if series and name != series:
            continue
        if len(published) > 1:
            raise ValueError(f"Conflicting published contract dates for {name}.")
        # The open-data CSV lacks the contract date. Never guess a holiday-adjusted expiry.
        if published:
            exp = next(iter(published))
        elif series and expiry:
            exp = expiry
        else:
            continue
        if exp > day and (expiry is None or exp == expiry):
            candidates.append((exp, name))
    if not candidates:
        raise ValueError("No matching future maturity. For an undated CSV supply both --series and --expiry.")
    target = min((exp for exp, _ in candidates), key=lambda e: (abs((e-day).days-14), e))
    names = sorted(name for exp, name in candidates if exp == target)
    if len(names) != 1:
        raise ValueError("Multiple series share this expiry. Supply --series.")
    chosen = names[0]
    date_basis = "official_contract_date" if dates[chosen] else "explicit_series_expiry"
    return chosen, target, date_basis


def clocks(day, expiry):
    observation = datetime.combine(day, time(13, 45), TAIPEI)
    end = datetime.combine(expiry, time(13, 30), TAIPEI)
    elapsed = end - observation
    if elapsed <= timedelta(0):
        raise ValueError("Maturity must follow the regular session close.")
    return observation, end, F(elapsed.days * 86400 + elapsed.seconds, 365 * 86400)


def quote_side(row):
    return SIDES.get(row["side"].lower())


def book_value(row, column):
    raw = row[column].replace(",", "")
    if raw in EMPTY:
        return None
    try:
        return F(raw)
    except (ValueError, ZeroDivisionError):
        return None


def prepare(raw, index, settings):
    day = date.fromisoformat(settings["date"])
    requested = date.fromisoformat(settings["expiry"]) if settings["expiry"] else None
    rows = read_day(raw, day)
    series, expiry, basis = choose_series(rows, day, requested, settings["series"])
    selected = [r for r in rows if r["series"] == series]
    spot = close_from_twse(index, day)
    observed, end, t = clocks(day, expiry)
    entries = []
    for row in selected:
        k = book_value(row, "strike")
        side = quote_side(row)
        if k is None or k <= 0 or side is None:
            raise ValueError("Invalid contract key in the selected TXO series.")
        b, a = book_value(row, "bid"), book_value(row, "ask")
        halted = row["halt"].lower() not in NO_HALT
        entries.append((k, side, b, a, halted))
    occurrences = Counter((k, side) for k, side, *_ in entries)
    groups = defaultdict(dict)
    rejected = Counter()
    evidence = []
    rejected_rows = []
    for k, side, bid, ask, halted in sorted(entries, key=lambda e: (e[0], e[1], str(e[2:]))):
        label = f"TXO:{series}:{side}:{k}"
        evidence.append([str(k), side, str(bid) if bid is not None else None,
                         str(ask) if ask is not None else None, halted])
        if occurrences[k, side] != 1:
            reason = "duplicate_contract"
        elif halted:
            reason = "trading_halt"
        elif bid is None or ask is None:
            reason = "missing_bid_or_ask"
        elif bid <= 0 or ask <= 0:
            reason = "nonpositive_bid_or_ask"
        elif ask < bid:
            reason = "crossed_market"
        else:
            groups[k][side] = Book(label, k, bid, ask)
            continue
        rejected[reason] += 1
        rejected_rows.append([label, reason])
    pairs = []
    for k, legs in sorted(groups.items()):
        if "C" in legs and "P" in legs:
            pairs.append(Pair(k, legs["C"], legs["P"]))
        else:
            rejected["unpaired_valid_leg"] += len(legs)
            rejected_rows.extend([leg.id, "unpaired_valid_leg"] for leg in legs.values())
    panel = [day.isoformat(), series, expiry.isoformat(), str(spot), evidence]
    key = sha256(json.dumps(panel, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    audit = dict(date=day.isoformat(), series=series, expiry=expiry.isoformat(), expiry_basis=basis,
                 valuation_taipei=observed.isoformat(), fixing_end_taipei=end.isoformat(), act365f=str(t),
                 regular_rows_at_expiry=len(entries), eligible_pairs=len(pairs), index_close=str(spot),
                 rejection_counts=dict(sorted(rejected.items())), rejected_rows=rejected_rows,
                 panel_sha256=key, source_contract_spec=SPEC)
    return pairs, spot, t, audit
