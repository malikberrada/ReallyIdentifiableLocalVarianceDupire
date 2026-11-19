"""Regenerate the small analytical control chains from their model parameters."""
from fractions import Fraction as F
from pathlib import Path

from dupire_identify.interval import black_call
from dupire_identify.io import Quote, write_quotes
from dupire_identify.workflow import synthetic_quotes


def informative_quotes():
    variance, spread = F(249999999, 10**9), F(1, 10**8)
    result = []
    for time in (F(1, 4), F(1)):
        for strike in map(F, ("13/20", "4/5", "47/50", "1", "21/20", "6/5", "3/2")):
            price = black_call(strike, time * variance)
            result.append(Quote(f"{time}_{strike}", time, strike, price.lo - spread, price.hi + spread))
    return result


def generate(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    synthetic_quotes(out / "r3-quotes.csv", backend="numpy")
    write_quotes(out / "informative-quotes.csv", informative_quotes())
