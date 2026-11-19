from dataclasses import dataclass
from fractions import Fraction as F
import csv
import hashlib
import json
from pathlib import Path
import os


def rational(x):
    return x if isinstance(x, F) else F(str(x))


@dataclass(frozen=True)
class Quote:
    id: str
    t: F
    k: F
    bid: F
    ask: F
    style: str = "european"


def read_quotes(path, european=True):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Aucune cotation dans le fichier.")
    result = []
    ids = set()
    for i, row in enumerate(rows):
        if not all(key in row for key in ("t", "k", "bid", "ask", "style")):
            raise ValueError("Colonnes requises: id,t,k,bid,ask,style (prix normalises).")
        q = Quote(row.get("id", str(i)), *(rational(row[key]) for key in ("t", "k", "bid", "ask")),
                  row["style"].lower())
        if q.id in ids:
            raise ValueError(f"Identifiant duplique: {q.id}")
        ids.add(q.id)
        if q.t <= 0 or q.k <= 0 or q.bid < 0 or q.ask < q.bid:
            raise ValueError(f"Cotation invalide: {q.id}")
        if european and q.style != "european":
            raise ValueError(f"{q.id}: P1/P2 requierent des options europeennes explicites.")
        if q.ask < max(0, 1-q.k) or q.bid > 1:
            raise ValueError(f"{q.id}: prix hors bornes normalisees.")
        result.append(q)
    return result


def write_quotes(path, quotes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "t", "k", "bid", "ask", "style"])
        for q in quotes:
            writer.writerow([q.id, str(q.t), str(q.k), str(q.bid), str(q.ask), q.style])


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    def default(x):
        if isinstance(x, F): return str(x)
        if hasattr(x, "tolist"): return x.tolist()
        raise TypeError(type(x).__name__)
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=default,
                               allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    os.replace(temp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def quote_payload(quotes):
    return [dict(id=q.id, t=str(q.t), k=str(q.k), bid=str(q.bid), ask=str(q.ask),
                 style=q.style) for q in quotes]


def payload_quotes(rows):
    return [Quote(r["id"], *(F(r[k]) for k in ("t", "k", "bid", "ask")), r["style"])
            for r in rows]
