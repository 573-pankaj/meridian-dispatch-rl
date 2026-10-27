#!/usr/bin/env python3
"""
greedy_jit.py - reference implementation of the Greedy-JIT baseline dispatcher
for the Meridian Air Network scenario (see meridian_scenario.md, section 9).

You are given this so you do not spend time re-implementing the baseline.
Use it as-is, import it, or read it. Python 3.9+, standard library only.

Command line
------------
  # Fixed instance -> schedule + decision log + KPIs
  python greedy_jit.py --requests requests.csv \
      --schedule greedy_schedule.csv --decisions greedy_decisions.csv

  # One generated day
  python greedy_jit.py --generate --seed 7

  # Many generated days (seeds 1000..1199), per-day KPIs to CSV
  python greedy_jit.py --generate --days 200 --seed 1000 --fleet 9 \
      --charge-mult 1.5 --kpis kpis.csv --jobs 4

  Other knobs: --pads 3,2,2,1,1   --demand-mult 1.5   --peak-shift 60

Python API
----------
  from greedy_jit import Config, load_requests, generate_requests, run_day
  cfg = Config(fleet_size=8)
  result = run_day(load_requests("requests.csv"), cfg)
  result.kpis, result.schedule, result.decisions

Interpretation of the scenario used by this baseline
-----------------------------------------------------
Where meridian_scenario.md leaves room for interpretation, this file is the
authoritative definition of Greedy-JIT. Your own environment / validator may
interpret the *scenario* differently, but say so if it does.

  1. Requests are processed in order of (earliest_dep, id). A request is
     revealed at max(0, earliest_dep - 30) and decided immediately.
  2. No activity in a new plan may start before the request's reveal time or
     before the aircraft's previous commitment ends. Committed plans are never
     changed.
  3. An aircraft is parked (off-pad) between commitments. Initially all
     aircraft are parked at t = 0 with 90% SoC.
  4. Direct case (aircraft already at origin): one pad stint [D - L, D) with
     L = max(5, minimum whole minutes to charge to 20 + energy(leg)).
  5. Deadhead case: a stint at the current vertiport of exactly the length
     needed for the deadhead (min 5), the deadhead flight, then a landing
     stint at the origin that lasts until the revenue departure D (min 5, and
     long enough to reach 20 + energy(revenue leg)). The deadhead departs as
     late as possible; if pads block that, it departs earlier, minute by
     minute.
  6. After every revenue landing: one 5-minute stint, then parked. If the
     next pre-departure stint starts exactly when that stint ends (same
     vertiport), the aircraft never parked, so the two are output as ONE stint
     (scenario section 5). Pad usage and charging are identical either way.
  7. For each aircraft, D is the earliest minute in [earliest_dep, latest_dep]
     for which the whole plan is feasible (pads, reserve, day end).
  8. Selection: minimum (deadhead km, D, aircraft index). If no aircraft is
     feasible, the request is rejected. Greedy-JIT therefore never has an
     accepted-then-failed request.
  9. Pad capacity is checked per minute over each stint's [start, end).
 10. Day end: the post-landing stint of every revenue flight must end <= 360.
 11. Fleet sizes other than 8: initial positions cycle through
     V0, V0, V1, V1, V2, V2, V3, V4.
 12. --charge-mult scales both charging rates (2.0 and 2/3 %/min).
 13. SoC is a float; it is never rounded internally. Output files round to
     2 decimals.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Scenario constants (meridian_scenario.md sections 1-7)
# --------------------------------------------------------------------------
VERTIPORTS = ["V0", "V1", "V2", "V3", "V4"]
XY = [(0, 0), (18, -6), (-9, 7), (12, 14), (-4, 24)]
DIST = [[round(math.hypot(a[0] - b[0], a[1] - b[1]), 1) for b in XY] for a in XY]
DEFAULT_PADS = (3, 2, 2, 1, 1)
INITIAL_POSITIONS = (0, 0, 1, 1, 2, 2, 3, 4)
DAY_END = 360
MIN_STINT = 5
RESERVE = 20.0
LEAD_TIME = 30
PRICE_PER_KM = 1.50
PRICE_PER_KWH = 0.30
ORIGIN_WEIGHTS = (1.6, 1.8, 1.0, 1.1, 0.7)


def block_time(d_km: float) -> int:
    """4 + ceil(d/3) minutes, computed in integer tenths of a km to avoid
    float artefacts (e.g. 30.0 / 3)."""
    tenths = int(round(d_km * 10))
    return 4 + (-(-tenths // 30))


def energy(d_km: float) -> float:
    """SoC percentage points consumed by a flight of d_km."""
    return 12.0 + 1.0 * d_km


# --------------------------------------------------------------------------
# Configuration and data types
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    fleet_size: int = 8
    pads: Tuple[int, ...] = DEFAULT_PADS
    charge_mult: float = 1.0
    initial_soc: float = 90.0

    def initial_fleet(self) -> List[Tuple[int, int, float]]:
        """(vertiport index, free-from time, SoC) per aircraft."""
        return [(INITIAL_POSITIONS[i % len(INITIAL_POSITIONS)], 0, self.initial_soc)
                for i in range(self.fleet_size)]


@dataclass(frozen=True)
class Request:
    id: str
    origin: int
    dest: int
    earliest_dep: int
    latest_dep: int
    dist_km: float
    revenue: float

    @property
    def reveal(self) -> int:
        return max(0, self.earliest_dep - LEAD_TIME)


@dataclass
class Activity:
    aircraft: int
    type: str            # "STINT" or "FLIGHT"
    frm: int             # vertiport index
    to: Optional[int]    # vertiport index for FLIGHT, None for STINT
    start: int
    end: int
    request_id: str = ""  # empty for deadheads and stints
    soc_start: float = 0.0
    soc_end: float = 0.0


@dataclass
class Plan:
    aircraft: int
    departure: int
    deadhead_km: float
    activities: List[Activity]
    new_state: Tuple[int, int, float]
    soc_at_departure: float
    min_landing_soc: float


@dataclass
class DayResult:
    kpis: dict
    schedule: List[Activity]
    decisions: List[dict]


# --------------------------------------------------------------------------
# Charging
# --------------------------------------------------------------------------
def charge(soc: float, minutes: float, mult: float = 1.0) -> float:
    """SoC after `minutes` on a pad (2 %/min below 80%, 2/3 %/min above, cap 100)."""
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
    """Shortest whole-minute stint (>= MIN_STINT) reaching `target` SoC, or None."""
    if target > 100.0 + 1e-9:
        return None
    if soc >= target:
        return MIN_STINT
    r1, r2 = 2.0 * mult, (2.0 / 3.0) * mult
    t = 0.0
    s = soc
    if s < 80.0:
        t1 = (min(target, 80.0) - s) / r1
        t += t1
        s = min(target, 80.0)
    if target > s:
        t += (target - s) / r2
    L = max(MIN_STINT, math.ceil(t - 1e-9))
    while charge(soc, L, mult) < target - 1e-9:  # guard against float edge cases
        L += 1
    return L


# --------------------------------------------------------------------------
# Pad occupancy
# --------------------------------------------------------------------------
class PadBook:
    def __init__(self, pads: Sequence[int]):
        self.cap = list(pads)
        self.occ = [[0] * DAY_END for _ in pads]

    def free(self, v: int, a: int, b: int) -> bool:
        if a < 0 or b > DAY_END or b <= a:
            return False
        if self.cap[v] <= 0:
            return False
        return max(self.occ[v][a:b]) < self.cap[v]

    def book(self, v: int, a: int, b: int) -> None:
        row = self.occ[v]
        for t in range(a, b):
            row[t] += 1


# --------------------------------------------------------------------------
# Planning one request for one aircraft
# --------------------------------------------------------------------------
def plan_for_aircraft(k: int, state: Tuple[int, int, float], req: Request,
                      pads: PadBook, cfg: Config) -> Optional[Plan]:
    loc, free_from, soc = state
    m = cfg.charge_mult
    start = max(free_from, req.reveal)
    o, d = req.origin, req.dest
    e_req = energy(req.dist_km)
    bt_req = block_time(req.dist_km)

    if loc == o:
        L = min_stint(soc, RESERVE + e_req, m)
        if L is None:
            return None
        soc_dep = charge(soc, L, m)
        for D in range(max(req.earliest_dep, start + L), req.latest_dep + 1):
            S = D - L
            arr = D + bt_req
            if arr + MIN_STINT > DAY_END:
                break
            if not pads.free(o, S, D) or not pads.free(d, arr, arr + MIN_STINT):
                continue
            soc_land = soc_dep - e_req
            soc_end = charge(soc_land, MIN_STINT, m)
            acts = [
                Activity(k, "STINT", o, None, S, D, "", soc, soc_dep),
                Activity(k, "FLIGHT", o, d, D, arr, req.id, soc_dep, soc_land),
                Activity(k, "STINT", d, None, arr, arr + MIN_STINT, "", soc_land, soc_end),
            ]
            return Plan(k, D, 0.0, acts, (d, arr + MIN_STINT, soc_end), soc_dep, soc_land)
        return None

    # Deadhead loc -> o, then revenue o -> d
    dh = DIST[loc][o]
    bt_dh = block_time(dh)
    e_dh = energy(dh)
    L1 = min_stint(soc, RESERVE + e_dh, m)
    if L1 is None:
        return None
    soc_dep1 = charge(soc, L1, m)
    soc_land1 = soc_dep1 - e_dh
    L2 = min_stint(soc_land1, RESERVE + e_req, m)
    if L2 is None:
        return None
    for D in range(req.earliest_dep, req.latest_dep + 1):
        arr = D + bt_req
        if arr + MIN_STINT > DAY_END:
            break
        if not pads.free(d, arr, arr + MIN_STINT):
            continue
        d1_latest = D - L2 - bt_dh
        d1_earliest = start + L1
        for D1 in range(d1_latest, d1_earliest - 1, -1):
            S1 = D1 - L1
            A = D1 + bt_dh
            if not pads.free(loc, S1, D1) or not pads.free(o, A, D):
                continue
            soc_dep = charge(soc_land1, D - A, m)
            soc_land = soc_dep - e_req
            soc_end = charge(soc_land, MIN_STINT, m)
            acts = [
                Activity(k, "STINT", loc, None, S1, D1, "", soc, soc_dep1),
                Activity(k, "FLIGHT", loc, o, D1, A, "", soc_dep1, soc_land1),
                Activity(k, "STINT", o, None, A, D, "", soc_land1, soc_dep),
                Activity(k, "FLIGHT", o, d, D, arr, req.id, soc_dep, soc_land),
                Activity(k, "STINT", d, None, arr, arr + MIN_STINT, "", soc_land, soc_end),
            ]
            return Plan(k, D, dh, acts, (d, arr + MIN_STINT, soc_end),
                        soc_dep, min(soc_land1, soc_land))
    return None


# --------------------------------------------------------------------------
# Running a day
# --------------------------------------------------------------------------
def run_day(requests: Sequence[Request], cfg: Config = Config()) -> DayResult:
    pads = PadBook(cfg.pads)
    fleet = cfg.initial_fleet()
    schedule: List[Activity] = []
    decisions: List[dict] = []
    revenue = km = km_dh = kwh = 0.0
    served = 0
    min_soc = math.inf

    for req in sorted(requests, key=lambda r: (r.earliest_dep, r.id)):
        best: Optional[Plan] = None
        for k, state in enumerate(fleet):
            p = plan_for_aircraft(k, state, req, pads, cfg)
            if p is None:
                continue
            if best is None or (p.deadhead_km, p.departure, p.aircraft) < \
                    (best.deadhead_km, best.departure, best.aircraft):
                best = p
        if best is None:
            decisions.append(dict(request_id=req.id, reveal_time=req.reveal,
                                  decision="REJECT", aircraft="", deadhead_from="",
                                  deadhead_km="", departure="", arrival="",
                                  soc_at_departure="", min_landing_soc=""))
            continue

        for a in best.activities:
            if a.type == "STINT":
                pads.book(a.frm, a.start, a.end)
        schedule.extend(best.activities)
        fleet[best.aircraft] = best.new_state
        served += 1
        revenue += req.revenue
        km += req.dist_km + best.deadhead_km
        km_dh += best.deadhead_km
        kwh += energy(req.dist_km) + (energy(best.deadhead_km) if best.deadhead_km > 0 else 0.0)
        min_soc = min(min_soc, best.min_landing_soc)
        rev_flight = next(a for a in best.activities if a.request_id == req.id)
        decisions.append(dict(
            request_id=req.id, reveal_time=req.reveal, decision="ACCEPT",
            aircraft=f"A{best.aircraft}",
            deadhead_from=VERTIPORTS[best.activities[0].frm] if best.deadhead_km > 0 else "",
            deadhead_km=best.deadhead_km, departure=rev_flight.start, arrival=rev_flight.end,
            soc_at_departure=round(best.soc_at_departure, 2),
            min_landing_soc=round(best.min_landing_soc, 2)))

    profit = revenue - PRICE_PER_KM * km - PRICE_PER_KWH * kwh
    kpis = dict(requests=len(requests), served=served, rejected=len(requests) - served,
                failures=0, revenue=round(revenue, 2), km_total=round(km, 1),
                km_deadhead=round(km_dh, 1), kwh=round(kwh, 1), profit=round(profit, 2),
                min_landing_soc=round(min_soc, 2) if served else "")
    for v, name in enumerate(VERTIPORTS):
        cap = cfg.pads[v]
        used = sum(pads.occ[v])
        kpis[f"pad_util_{name}"] = round(used / (cap * DAY_END), 3) if cap > 0 else ""
    schedule.sort(key=lambda a: (a.aircraft, a.start, a.type != "STINT"))
    return DayResult(kpis, _merge_adjacent_stints(schedule), decisions)


def _merge_adjacent_stints(schedule: List[Activity]) -> List[Activity]:
    """A landing stint immediately followed by a pre-departure stint at the same
    vertiport means the aircraft never parked, so per scenario section 5 it is one
    stint. Charging is unaffected (charge(charge(s, a), b) == charge(s, a + b))."""
    out: List[Activity] = []
    for a in schedule:
        prev = out[-1] if out else None
        if (prev is not None and a.type == "STINT" and prev.type == "STINT"
                and prev.aircraft == a.aircraft and prev.frm == a.frm and prev.end == a.start):
            prev.end = a.end
            prev.soc_end = a.soc_end
        else:
            out.append(a)
    return out


# --------------------------------------------------------------------------
# Demand: loading and generation (meridian_scenario.md section 6)
# --------------------------------------------------------------------------
def load_requests(path: str) -> List[Request]:
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.append(Request(r["id"], VERTIPORTS.index(r["origin"]),
                               VERTIPORTS.index(r["dest"]), int(r["earliest_dep"]),
                               int(r["latest_dep"]), float(r["dist_km"]), float(r["revenue"])))
    return out


def demand_rate(t: float, demand_mult: float = 1.0, peak_shift: float = 0.0) -> float:
    return demand_mult * 0.2 * (1.0 + 1.4 * math.exp(-((t - 150.0 - peak_shift) / 50.0) ** 2))


def generate_requests(seed: int, demand_mult: float = 1.0,
                      peak_shift: float = 0.0) -> List[Request]:
    """Non-homogeneous Poisson arrivals on [0, 300) by thinning. Not capped at 70."""
    rng = random.Random(seed)
    lam_max = demand_mult * 0.2 * 2.4
    t, raw = 0.0, []
    while True:
        t += rng.expovariate(lam_max)
        if t >= 300.0:
            break
        if rng.random() * lam_max <= demand_rate(t, demand_mult, peak_shift):
            o = rng.choices(range(5), ORIGIN_WEIGHTS)[0]
            others = [j for j in range(5) if j != o]
            d = rng.choices(others, [ORIGIN_WEIGHTS[j] for j in others])[0]
            raw.append((int(t), o, d))
    out = []
    width = max(2, len(str(len(raw) - 1)))
    for i, (e, o, d) in enumerate(raw):
        dist = DIST[o][d]
        out.append(Request(f"R{i:0{width}d}", o, d, e, e + 15, dist, float(round(40 + 4 * dist))))
    return out


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def write_schedule(path: str, schedule: Sequence[Activity]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aircraft", "type", "from", "to", "start", "end", "request_id"])
        for a in schedule:
            w.writerow([f"A{a.aircraft}", a.type, VERTIPORTS[a.frm],
                        VERTIPORTS[a.to] if a.to is not None else "",
                        a.start, a.end, a.request_id])


def write_dicts(path: str, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _run_seed(args):
    seed, cfg, demand_mult, peak_shift = args
    reqs = generate_requests(seed, demand_mult, peak_shift)
    return dict(seed=seed, **run_day(reqs, cfg).kpis)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Greedy-JIT baseline for the Meridian scenario.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--requests", help="requests CSV (scenario section 8 format)")
    src.add_argument("--generate", action="store_true", help="sample day(s) from the generator")
    p.add_argument("--seed", type=int, default=0, help="seed (first seed if --days > 1)")
    p.add_argument("--days", type=int, default=1, help="number of generated days")
    p.add_argument("--fleet", type=int, default=8)
    p.add_argument("--pads", default=",".join(map(str, DEFAULT_PADS)),
                   help="pad counts V0..V4, e.g. 3,2,2,1,1")
    p.add_argument("--charge-mult", type=float, default=1.0)
    p.add_argument("--demand-mult", type=float, default=1.0, help="generator only")
    p.add_argument("--peak-shift", type=float, default=0.0, help="generator only, minutes")
    p.add_argument("--schedule", help="write schedule CSV (single day only)")
    p.add_argument("--decisions", help="write decision log CSV (single day only)")
    p.add_argument("--kpis", help="write per-day KPI CSV")
    p.add_argument("--jobs", type=int, default=1, help="worker processes for --days > 1")
    a = p.parse_args(argv)

    pads = tuple(int(x) for x in a.pads.split(","))
    if len(pads) != 5:
        p.error("--pads needs 5 comma-separated integers")
    cfg = Config(fleet_size=a.fleet, pads=pads, charge_mult=a.charge_mult)

    if a.requests or a.days == 1:
        reqs = load_requests(a.requests) if a.requests else \
            generate_requests(a.seed, a.demand_mult, a.peak_shift)
        res = run_day(reqs, cfg)
        if a.schedule:
            write_schedule(a.schedule, res.schedule)
        if a.decisions:
            write_dicts(a.decisions, res.decisions)
        if a.kpis:
            write_dicts(a.kpis, [res.kpis])
        for k, v in res.kpis.items():
            print(f"{k:>16}: {v}")
        return 0

    if a.schedule or a.decisions:
        p.error("--schedule/--decisions are for a single day; use --kpis with --days")
    tasks = [(s, cfg, a.demand_mult, a.peak_shift) for s in range(a.seed, a.seed + a.days)]
    if a.jobs > 1:
        from multiprocessing import Pool
        with Pool(a.jobs) as pool:
            rows = pool.map(_run_seed, tasks)
    else:
        rows = [_run_seed(t) for t in tasks]
    if a.kpis:
        write_dicts(a.kpis, rows)
    for key in ("profit", "served", "requests", "km_deadhead"):
        xs = [float(r[key]) for r in rows]
        mu = statistics.fmean(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        print(f"{key:>12}: mean {mu:9.2f}  sd {sd:8.2f}  (n={len(xs)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
