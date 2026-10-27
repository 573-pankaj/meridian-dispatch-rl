"""Event-driven dispatch environment for the Meridian scenario.

One decision epoch per request, in order of (earliest_dep, id). At each epoch
the environment builds a set of *candidate plans* (aircraft k, charging mode m).
Every candidate is fully checked before it is offered (pads, reserve, day end,
reveal time, aircraft availability), so any action the policy can pick is
feasible. Infeasible or duplicate candidates are masked. Action 0 = REJECT is
always available.

Action index layout (Gym style): a = 0 is REJECT, a = 1 + m * K + k is
"aircraft k with mode m", where K is the fleet size.

The JIT mode reproduces baseline/greedy_jit.py plan_for_aircraft exactly
(checked in tests/test_suite.py). Other modes only change how long the aircraft
charges or when it departs; they never relax a constraint.
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from .physics import (DAY_END, DIST, DEFAULT_PADS, FAST_LIMIT, INITIAL_POSITIONS,
                      LEAD_TIME, MIN_STINT, ORIGIN_WEIGHTS, PRICE_PER_KM,
                      PRICE_PER_KWH, RESERVE, VERTIPORTS, PadBook, block_time,
                      charge, energy, flight_cost, min_stint)


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    fleet_size: int = 8
    pads: Tuple[int, ...] = DEFAULT_PADS
    charge_mult: float = 1.0
    initial_soc: float = 90.0

    def initial_fleet(self) -> List[Tuple[int, int, float]]:
        return [(INITIAL_POSITIONS[i % len(INITIAL_POSITIONS)], 0, self.initial_soc)
                for i in range(self.fleet_size)]

    def to_dict(self) -> dict:
        return dict(fleet_size=self.fleet_size, pads=list(self.pads),
                    charge_mult=self.charge_mult, initial_soc=self.initial_soc)


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

    @property
    def direct_margin(self) -> float:
        return self.revenue - flight_cost(self.dist_km)


@dataclass
class Activity:
    aircraft: int
    type: str            # "STINT" or "FLIGHT"
    frm: int
    to: Optional[int]
    start: int
    end: int
    request_id: str = ""
    soc_start: float = 0.0
    soc_end: float = 0.0


class Mode(NamedTuple):
    name: str
    pre_extend: bool             # start the pre-departure stint earlier, if pads allow
    post_target: Optional[float]  # keep charging after landing up to this SoC (None = 5 min)
    late: bool                   # latest feasible departure instead of earliest


JIT = Mode("JIT", False, None, False)
TOPUP = Mode("TOPUP", False, FAST_LIMIT, False)
PRECHARGE = Mode("PRECHARGE", True, None, False)
LATE = Mode("LATE", False, None, True)
ALL_MODES: Tuple[Mode, ...] = (JIT, TOPUP, PRECHARGE, LATE)
PRE_TARGET = FAST_LIMIT  # pre-extension charges towards max(required, 80%)


@dataclass
class Plan:
    aircraft: int
    mode: int
    departure: int
    arrival: int
    deadhead_km: float
    deadhead_from: int
    activities: List[Activity]
    new_state: Tuple[int, int, float]
    soc_at_departure: float
    min_landing_soc: float
    profit: float           # revenue - revenue-flight cost - deadhead cost
    first_start: int        # first minute the plan uses
    pad_minutes: Dict[int, int] = field(default_factory=dict)

    def key(self):
        return tuple((a.type, a.frm, a.to, a.start, a.end) for a in self.activities)


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------
def _post_stint(pads: PadBook, d: int, arr: int, soc_land: float, mode: Mode,
                mult: float) -> int:
    """Length of the stint after landing. JIT: 5 min. TOPUP: extend minute by
    minute while a pad stays free, until the post target or day end is reached."""
    L = MIN_STINT
    if mode.post_target is None:
        return L
    Lt = min_stint(soc_land, mode.post_target, mult) or MIN_STINT
    Lmax = min(Lt, DAY_END - arr)
    while L < Lmax and pads.free_minute(d, arr + L):
        L += 1
    return L


def _extend_back(pads: PadBook, v: int, S: int, D: int, lo: int, soc: float,
                 target: float, mult: float) -> int:
    """Move the start of a stint [S, D) earlier (not below lo) while a pad is free,
    until the stint is long enough to reach `target`."""
    Lt = min_stint(soc, target, mult)
    if Lt is None:
        Lt = min_stint(soc, 100.0, mult)
    s_min = max(lo, D - Lt)
    while S - 1 >= s_min and pads.free_minute(v, S - 1):
        S -= 1
    return S


def plan_for_aircraft(k: int, state: Tuple[int, int, float], req: Request,
                      pads: PadBook, cfg: Config, mode: Mode = JIT,
                      mode_index: int = 0) -> Optional[Plan]:
    loc, free_from, soc = state
    m = cfg.charge_mult
    start = max(free_from, req.reveal)
    o, d = req.origin, req.dest
    e_req = energy(req.dist_km)
    bt_req = block_time(req.dist_km)
    rev_profit = req.revenue - flight_cost(req.dist_km)

    if loc == o:
        L = min_stint(soc, RESERVE + e_req, m)
        if L is None:
            return None
        Ds = range(max(req.earliest_dep, start + L), req.latest_dep + 1)
        if mode.late:
            Ds = reversed(Ds)
        for D in Ds:
            S = D - L
            arr = D + bt_req
            if arr + MIN_STINT > DAY_END:
                if mode.late:
                    continue
                break
            if not pads.free(o, S, D) or not pads.free(d, arr, arr + MIN_STINT):
                continue
            if mode.pre_extend:
                S = _extend_back(pads, o, S, D, start, soc,
                                 max(RESERVE + e_req, PRE_TARGET), m)
            soc_dep = charge(soc, D - S, m)
            soc_land = soc_dep - e_req
            Lp = _post_stint(pads, d, arr, soc_land, mode, m)
            soc_end = charge(soc_land, Lp, m)
            acts = [
                Activity(k, "STINT", o, None, S, D, "", soc, soc_dep),
                Activity(k, "FLIGHT", o, d, D, arr, req.id, soc_dep, soc_land),
                Activity(k, "STINT", d, None, arr, arr + Lp, "", soc_land, soc_end),
            ]
            return Plan(k, mode_index, D, arr, 0.0, -1, acts, (d, arr + Lp, soc_end),
                        soc_dep, soc_land, rev_profit, S, _pad_minutes(acts))
        return None

    # Deadhead loc -> o, then revenue o -> d
    dh = DIST[loc][o]
    bt_dh = block_time(dh)
    e_dh = energy(dh)
    L1 = min_stint(soc, RESERVE + e_dh, m)
    if L1 is None:
        return None
    soc_land1_jit = charge(soc, L1, m) - e_dh
    L2 = min_stint(soc_land1_jit, RESERVE + e_req, m)
    if L2 is None:
        return None
    Ds = range(req.earliest_dep, req.latest_dep + 1)
    if mode.late:
        Ds = reversed(Ds)
    for D in Ds:
        arr = D + bt_req
        if arr + MIN_STINT > DAY_END:
            if mode.late:
                continue
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
            if mode.pre_extend:
                S1 = _extend_back(pads, loc, S1, D1, start, soc,
                                  max(RESERVE + e_dh, PRE_TARGET), m)
            soc_dep1 = charge(soc, D1 - S1, m)
            soc_land1 = soc_dep1 - e_dh
            soc_dep = charge(soc_land1, D - A, m)
            soc_land = soc_dep - e_req
            Lp = _post_stint(pads, d, arr, soc_land, mode, m)
            soc_end = charge(soc_land, Lp, m)
            acts = [
                Activity(k, "STINT", loc, None, S1, D1, "", soc, soc_dep1),
                Activity(k, "FLIGHT", loc, o, D1, A, "", soc_dep1, soc_land1),
                Activity(k, "STINT", o, None, A, D, "", soc_land1, soc_dep),
                Activity(k, "FLIGHT", o, d, D, arr, req.id, soc_dep, soc_land),
                Activity(k, "STINT", d, None, arr, arr + Lp, "", soc_land, soc_end),
            ]
            return Plan(k, mode_index, D, arr, dh, loc, acts, (d, arr + Lp, soc_end),
                        soc_dep, min(soc_land1, soc_land), rev_profit - flight_cost(dh),
                        S1, _pad_minutes(acts))
    return None


def _pad_minutes(acts: Sequence[Activity]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for a in acts:
        if a.type == "STINT":
            out[a.frm] = out.get(a.frm, 0) + a.end - a.start
    return out


# --------------------------------------------------------------------------
# Demand
# --------------------------------------------------------------------------
def load_requests(path: str) -> List[Request]:
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.append(Request(r["id"], VERTIPORTS.index(r["origin"]),
                               VERTIPORTS.index(r["dest"]), int(r["earliest_dep"]),
                               int(r["latest_dep"]), float(r["dist_km"]),
                               float(r["revenue"])))
    return out


def demand_rate(t: float, demand_mult: float = 1.0, peak_shift: float = 0.0) -> float:
    return demand_mult * 0.2 * (1.0 + 1.4 * math.exp(-((t - 150.0 - peak_shift) / 50.0) ** 2))


def generate_requests(seed: int, demand_mult: float = 1.0,
                      peak_shift: float = 0.0) -> List[Request]:
    """Same generator and random stream as baseline/greedy_jit.py (thinning).
    With the same seed it returns the same requests as the baseline."""
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
            dd = rng.choices(others, [ORIGIN_WEIGHTS[j] for j in others])[0]
            raw.append((int(t), o, dd))
    out = []
    width = max(2, len(str(len(raw) - 1)))
    for i, (e, o, dd) in enumerate(raw):
        dist = DIST[o][dd]
        out.append(Request(f"R{i:0{width}d}", o, dd, e, e + 15, dist,
                           float(round(40 + 4 * dist))))
    return out


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
class MeridianEnv:
    """Gym-style environment: reset() -> obs, step(action) -> (obs, reward, done, info).

    `obs` is a dict holding the raw state (request, fleet, pads, candidates, mask).
    Feature vectors are built by meridian.agent so that the env stays policy-free.
    """

    def __init__(self, cfg: Config = Config(), modes: Sequence[Mode] = ALL_MODES,
                 accept_bonus: float = 0.0):
        self.cfg = cfg
        self.modes = tuple(modes)
        self.accept_bonus = accept_bonus  # only used for the rejected reward variant
        self.n_actions = 1 + len(self.modes) * cfg.fleet_size

    # ---- episode control -------------------------------------------------
    def reset(self, requests: Optional[Sequence[Request]] = None, seed: Optional[int] = None,
              demand_mult: float = 1.0, peak_shift: float = 0.0) -> dict:
        if requests is None:
            if seed is None:
                raise ValueError("give either requests or seed")
            requests = generate_requests(seed, demand_mult, peak_shift)
        self.requests = sorted(requests, key=lambda r: (r.earliest_dep, r.id))
        self.pads = PadBook(self.cfg.pads)
        self.fleet = self.cfg.initial_fleet()
        self.schedule: List[Activity] = []
        self.decisions: List[dict] = []
        self.i = 0
        self.totals = dict(revenue=0.0, km=0.0, km_dh=0.0, kwh=0.0, served=0,
                           forced_reject=0, strategic_reject=0, min_soc=math.inf)
        self.mode_counts = [0] * len(self.modes)
        return self._observe()

    @property
    def done(self) -> bool:
        return self.i >= len(self.requests)

    def _observe(self) -> dict:
        if self.done:
            self.cands, self.mask = [], [True]
            return dict(done=True)
        req = self.requests[self.i]
        K = self.cfg.fleet_size
        cands: List[Optional[Plan]] = [None] * (len(self.modes) * K)
        seen = set()
        for mi, mode in enumerate(self.modes):
            for k in range(K):
                p = plan_for_aircraft(k, self.fleet[k], req, self.pads, self.cfg, mode, mi)
                if p is None:
                    continue
                key = (k, p.key())
                if key in seen:     # same plan as an earlier mode: mask the duplicate
                    continue
                seen.add(key)
                cands[mi * K + k] = p
        self.cands = cands
        self.mask = [True] + [c is not None for c in cands]
        return dict(done=False, t=req.reveal, request=req, fleet=list(self.fleet),
                    pads=self.pads, candidates=cands, mask=list(self.mask), step=self.i,
                    n_requests=len(self.requests))

    def step(self, action: int):
        if self.done:
            raise RuntimeError("episode finished")
        if not (0 <= action < self.n_actions) or not self.mask[action]:
            # Safety by construction: an unmasked action cannot be infeasible.
            raise ValueError(f"action {action} is masked (infeasible)")
        req = self.requests[self.i]
        reward = 0.0
        if action == 0:
            feasible = any(self.mask[1:])
            self.totals["strategic_reject" if feasible else "forced_reject"] += 1
            self.decisions.append(dict(request_id=req.id, reveal_time=req.reveal,
                                       decision="REJECT", reject_type="strategic" if feasible else "forced",
                                       aircraft="", mode="", deadhead_from="", deadhead_km="",
                                       departure="", arrival="", soc_at_departure="",
                                       min_landing_soc=""))
        else:
            p = self.cands[action - 1]
            self._commit(req, p)
            reward = p.profit + self.accept_bonus
        self.i += 1
        obs = self._observe()
        return obs, reward, self.done, {}

    def _commit(self, req: Request, p: Plan) -> None:
        for a in p.activities:
            if a.type == "STINT":
                self.pads.book(a.frm, a.start, a.end)
        self.schedule.extend(p.activities)
        self.fleet[p.aircraft] = p.new_state
        t = self.totals
        t["served"] += 1
        t["revenue"] += req.revenue
        t["km"] += req.dist_km + p.deadhead_km
        t["km_dh"] += p.deadhead_km
        t["kwh"] += energy(req.dist_km) + (energy(p.deadhead_km) if p.deadhead_km > 0 else 0.0)
        t["min_soc"] = min(t["min_soc"], p.min_landing_soc)
        self.mode_counts[p.mode] += 1
        self.decisions.append(dict(
            request_id=req.id, reveal_time=req.reveal, decision="ACCEPT", reject_type="",
            aircraft=f"A{p.aircraft}", mode=self.modes[p.mode].name,
            deadhead_from=VERTIPORTS[p.deadhead_from] if p.deadhead_km > 0 else "",
            deadhead_km=p.deadhead_km, departure=p.departure, arrival=p.arrival,
            soc_at_departure=round(p.soc_at_departure, 2),
            min_landing_soc=round(p.min_landing_soc, 2)))

    # ---- outputs ---------------------------------------------------------
    def merged_schedule(self) -> List[Activity]:
        sched = sorted((Activity(**vars(a)) for a in self.schedule),
                       key=lambda a: (a.aircraft, a.start, a.type != "STINT"))
        out: List[Activity] = []
        for a in sched:
            prev = out[-1] if out else None
            if (prev is not None and a.type == "STINT" and prev.type == "STINT"
                    and prev.aircraft == a.aircraft and prev.frm == a.frm
                    and prev.end == a.start):
                prev.end, prev.soc_end = a.end, a.soc_end
            else:
                out.append(a)
        return out

    def kpis(self) -> dict:
        t = self.totals
        profit = t["revenue"] - PRICE_PER_KM * t["km"] - PRICE_PER_KWH * t["kwh"]
        n = len(self.requests)
        out = dict(requests=n, served=t["served"], rejected=n - t["served"],
                   forced_reject=t["forced_reject"], strategic_reject=t["strategic_reject"],
                   failures=0, revenue=round(t["revenue"], 2), km_total=round(t["km"], 1),
                   km_deadhead=round(t["km_dh"], 1), kwh=round(t["kwh"], 1),
                   profit=round(profit, 2),
                   min_landing_soc=round(t["min_soc"], 2) if t["served"] else "")
        for v, name in enumerate(VERTIPORTS):
            cap = self.cfg.pads[v]
            out[f"pad_util_{name}"] = (round(sum(self.pads.occ[v]) / (cap * DAY_END), 3)
                                       if cap > 0 else "")
        for mi, mode in enumerate(self.modes):
            out[f"mode_{mode.name}"] = self.mode_counts[mi]
        return out


def jit_action(obs: dict, env: MeridianEnv) -> int:
    """Greedy-JIT as an env policy: JIT-mode candidates only, minimum
    (deadhead km, D, aircraft index); reject only if none is feasible."""
    K = env.cfg.fleet_size
    best, best_key = 0, None
    mi = env.modes.index(JIT)
    for k in range(K):
        p = obs["candidates"][mi * K + k]
        if p is None:
            continue
        key = (p.deadhead_km, p.departure, k)
        if best_key is None or key < best_key:
            best, best_key = 1 + mi * K + k, key
    return best


def run_policy(env: MeridianEnv, policy, **reset_kwargs) -> dict:
    """Run one day. `policy(obs, env) -> action`. Returns the env KPIs."""
    obs = env.reset(**reset_kwargs)
    while not env.done:
        obs, _, _, _ = env.step(policy(obs, env))
    return env.kpis()
