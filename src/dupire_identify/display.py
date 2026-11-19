"""Non-authoritative, JSON-safe previews of exact numbers.

Losing a floating-point preview must never lose an exact certificate. None
means that the exact value, saved separately as a rational string, is outside
the finite binary64 range. This function is NOT for solver coefficients.
"""
import math


def approximate(value):
    try:
        result = float(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None
