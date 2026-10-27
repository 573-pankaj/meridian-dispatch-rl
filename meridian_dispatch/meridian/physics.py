"""Scenario physics for the Meridian network (Scenario.pdf sections 1-7).

This module is written to match baseline/greedy_jit.py exactly (same rounding,
same float handling). tests/test_suite.py checks the match.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

VERTIPORTS = ["V0", "V1", "V2", "V3", "V4"]
XY = [(0, 0), (18, -6), (-9, 7), (12, 14), (-4, 24)]
DIST = [[round(math.hypot(a[0] - b[0], a[1] - b[1]), 1) for b in XY] for a in XY]
DEFAULT_PADS = (3, 2, 2, 1, 1)
INITIAL_POSITIONS = (0, 0, 1, 1, 2, 2, 3, 4)
ORIGIN_WEIGHTS = (1.6, 1.8, 1.0, 1.1, 0.7)
DAY_END = 360
MIN_STINT = 5
RESERVE = 20.0
LEAD_TIME = 30
PRICE_PER_KM = 1.50
PRICE_PER_KWH = 0.30
FAST_LIMIT = 80.0  # end of the fast-charging region


def block_time(d_km: float) -> int:
    """4 + ceil(d/3) minutes, in integer tenths of a km (30.0/3 must give 10, not 11)."""
    tenths = int(round(d_km * 10))
    return 4 + (-(-tenths // 30))


def energy(d_km: float) -> float:
    """SoC percentage points (= kWh) used by a flight of d_km."""
    return 12.0 + 1.0 * d_km


def flight_cost(d_km: float) -> float:
    """Distance cost plus energy cost of one flight (revenue or deadhead)."""
    return PRICE_PER_KM * d_km + PRICE_PER_KWH * energy(d_km)


def charge(soc: float, minutes: float, mult: float = 1.0) -> float:
    """SoC after `minutes` on a pad: 2 %/min below 80, 2/3 %/min above, cap 100."""
    r1, r2 = 2.0 * mult, (2.0 / 3.0) * mult
    t = float(minutes)
    if soc < 80.0 and t > 0:
        dt = min(t, (80.0 - soc) / r1)
        soc += r1 * dt
        t -= dt
    if t > 0 and soc < 100.0:
        soc = min(100.0, soc + r2 * t)
    return soc


def min_stint(soc: float, target: float, mult: float = 1.0) -> Optional[int]:
    """Shortest whole-minute stint (>= 5) that reaches `target`, or None if target > 100."""
    if target > 100.0 + 1e-9:
        return None
    if soc >= target:
        return MIN_STINT
    r1, r2 = 2.0 * mult, (2.0 / 3.0) * mult
    t, s = 0.0, soc
    if s < 80.0:
        t += (min(target, 80.0) - s) / r1
        s = min(target, 80.0)
    if target > s:
        t += (target - s) / r2
    L = max(MIN_STINT, math.ceil(t - 1e-9))
    while charge(soc, L, mult) < target - 1e-9:
        L += 1
    return L


class PadBook:
    """Minute-by-minute pad occupancy, intervals are [a, b)."""

    def __init__(self, pads: Sequence[int]):
        self.cap = list(pads)
        self.occ = [[0] * DAY_END for _ in pads]

    def free(self, v: int, a: int, b: int) -> bool:
        if a < 0 or b > DAY_END or b <= a or self.cap[v] <= 0:
            return False
        return max(self.occ[v][a:b]) < self.cap[v]

    def free_minute(self, v: int, t: int) -> bool:
        return 0 <= t < DAY_END and self.occ[v][t] < self.cap[v]

    def book(self, v: int, a: int, b: int) -> None:
        row = self.occ[v]
        for t in range(a, b):
            row[t] += 1

    def copy(self) -> "PadBook":
        pb = PadBook.__new__(PadBook)
        pb.cap = list(self.cap)
        pb.occ = [list(r) for r in self.occ]
        return pb

    def utilisation(self, v: int, a: int, b: int) -> float:
        a, b = max(0, a), min(DAY_END, b)
        if b <= a or self.cap[v] <= 0:
            return 1.0 if self.cap[v] <= 0 else 0.0
        return sum(self.occ[v][a:b]) / (self.cap[v] * (b - a))
