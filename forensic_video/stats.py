"""Numerically stable, deterministic robust statistics."""

from typing import Iterable, List, Optional
import math


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def median(values: Iterable[float]) -> Optional[float]:
    xs = sorted(finite(values))
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def mad(values: Iterable[float], center: Optional[float] = None) -> Optional[float]:
    xs = finite(values)
    if not xs:
        return None
    c = median(xs) if center is None else center
    return median(abs(x - c) for x in xs)


def robust_z(value: float, values: Iterable[float]) -> Optional[float]:
    c = median(values)
    scale = mad(values, c)
    if c is None or scale is None:
        return None
    return (value - c) / (1.4826 * scale + 1e-9)


def percentile(values: Iterable[float], fraction: float) -> Optional[float]:
    xs = sorted(finite(values))
    if not xs:
        return None
    q = min(1.0, max(0.0, fraction)) * (len(xs) - 1)
    lo, hi = int(q), min(len(xs) - 1, int(q) + 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (q - lo)


def clipped(value: Optional[float], low: float = 0.0, high: float = 1.0) -> Optional[float]:
    """Clip a finite statistic without turning missing evidence into zero."""
    if value is None or not math.isfinite(float(value)):
        return None
    return max(low, min(high, float(value)))


def outlier_fraction(values: Iterable[float], threshold: float = 3.5) -> float:
    """Fraction of robust-MAD outliers, with a stable zero-MAD fallback."""
    xs = finite(values)
    if len(xs) < 3:
        return 0.0
    center = median(xs)
    scale = mad(xs, center)
    if center is None or scale is None:
        return 0.0
    if scale < 1e-9:
        return sum(1 for value in xs if abs(value - center) > 1e-9) / len(xs)
    return sum(1 for value in xs if abs(value - center) / (1.4826 * scale) > threshold) / len(xs)
