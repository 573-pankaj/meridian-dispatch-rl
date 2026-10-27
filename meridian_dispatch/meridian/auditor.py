#!/usr/bin/env python3
"""Independent constraint checker for a completed Meridian episode.

This file imports nothing from meridian.env, meridian.physics or the baseline.
It re-derives the network, distances, block times, energy and charging from
Scenario.pdf and re-checks a finished day from its raw CSV logs only:

  python -m meridian.auditor --requests data/requests.csv \
      --schedule results/x_schedule.csv --decisions results/x_decisions.csv \
      [--config cfg.json] [--kpis results/x_kpis.csv]

Inputs
  schedule.csv : aircraft,type,from,to,start,end,request_id   (STINT / FLIGHT rows)
  decisions.csv: request_id,reveal_time,decision,...           (ACCEPT / REJECT)
  requests.csv : id,origin,dest,earliest_dep,latest_dep,dist_km,revenue
  config.json  : {"fleet_size":8,"pads":[3,2,2,1,1],"charge_mult":1.0,"initial_soc":90.0}

Exit code 0 if no violation is found, 1 otherwise.

Rule list (code: meaning)
  D1  every request has exactly one decision, with the correct reveal time
  F1  every ACCEPT has exactly one revenue flight; else it is a service FAILURE
  F2  a REJECTed or unknown request has no revenue flight
  F3  revenue flight: right origin/destination, departure inside the window
  T1  flight duration = 4 + ceil(d / 3) for the straight-line distance d
  T2  every activity lies inside [0, 360]; stints have length >= 5
  C1  first activity of each aircraft is at its initial vertiport
  C2  no overlapping activities; location is continuous (parked aircraft do not move)
  C3  every departure ends a stint at the origin; every landing starts a stint at the destination
  E1  SoC at every landing >= 20 %  (SoC rebuilt from 90 % at t = 0)
  E2  SoC never above 100 %
  P1  minute-by-minute pad occupancy <= pad count at every vertiport
  R1  activities that prepare a revenue flight start no earlier than the request's reveal time
  K1  KPIs recomputed from the schedule match the reported KPIs (if given)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict

# ---- scenario facts, typed in again from Scenario.pdf --------------------
_PORTS = {"V0": (0.0, 0.0), "V1": (18.0, -6.0), "V2": (-9.0, 7.0),
          "V3": (12.0, 14.0), "V4": (-4.0, 24.0)}
_START_PORT_CYCLE = ["V0", "V0", "V1", "V1", "V2", "V2", "V3", "V4"]
_DAY_MINUTES = 360
_TURNAROUND = 5
_MIN_LANDING_SOC = 20.0
_TOL = 1e-7


def _km(a: str, b: str) -> float:
    (x1, y1), (x2, y2) = _PORTS[a], _PORTS[b]
    return round(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2), 1)


def _flight_minutes(km: float) -> int:
    # ceil(km / 3) done on integer tenths, so 30.0 km gives exactly 10
    tenths = int(round(km * 10))
    return 4 + (tenths + 29) // 30


def _flight_energy(km: float) -> float:
    return 12.0 + km


def _after_charging(soc: float, minutes: int, speed: float) -> float:
    """Piecewise-linear charging: fast rate up to 80 %, slow rate to 100 %."""
    fast, slow = 2.0 * speed, 2.0 * speed / 3.0
    left = float(minutes)
    if soc < 80.0:
        need = (80.0 - soc) / fast
        if left <= need:
            return soc + fast * left
        soc, left = 80.0, left - need
    return min(100.0, soc + slow * left)


# ---- loading -------------------------------------------------------------
def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def audit(requests_rows, schedule_rows, decision_rows, config: dict,
          kpi_row: dict = None) -> dict:
    fleet = int(config.get("fleet_size", 8))
    pads = [int(x) for x in config.get("pads", [3, 2, 2, 1, 1])]
    speed = float(config.get("charge_mult", 1.0))
    soc0 = float(config.get("initial_soc", 90.0))
    pad_of = dict(zip(sorted(_PORTS), pads))
    violations = []

    def bad(code, msg):
        violations.append(f"{code}: {msg}")

    reqs = {}
    for r in requests_rows:
        reqs[r["id"]] = dict(o=r["origin"], d=r["dest"], e=int(r["earliest_dep"]),
                             l=int(r["latest_dep"]), km=float(r["dist_km"]),
                             rev=float(r["revenue"]))
        if abs(_km(r["origin"], r["dest"]) - float(r["dist_km"])) > 1e-9:
            bad("F3", f"{r['id']} dist_km {r['dist_km']} != geometry")

    # D1 -------------------------------------------------------------------
    dec = {}
    for r in decision_rows:
        rid = r["request_id"]
        if rid in dec:
            bad("D1", f"{rid} decided twice")
        dec[rid] = r
        if rid not in reqs:
            bad("D1", f"decision for unknown request {rid}")
            continue
        if int(r["reveal_time"]) != max(0, reqs[rid]["e"] - 30):
            bad("D1", f"{rid} reveal_time {r['reveal_time']} wrong")
        if r["decision"] not in ("ACCEPT", "REJECT"):
            bad("D1", f"{rid} unknown decision {r['decision']}")
    for rid in reqs:
        if rid not in dec:
            bad("D1", f"{rid} has no decision")

    # parse activities -------------------------------------------------------
    acts = defaultdict(list)
    for r in schedule_rows:
        name = r["aircraft"]
        if not name.startswith("A") or not (0 <= int(name[1:]) < fleet):
            bad("C1", f"unknown aircraft {name}")
            continue
        acts[int(name[1:])].append(dict(type=r["type"], frm=r["from"], to=r["to"] or None,
                                        s=int(r["start"]), e=int(r["end"]),
                                        rid=r.get("request_id", "") or ""))

    # F1/F2/F3 ----------------------------------------------------------------
    rev_flights = defaultdict(list)
    for k, lst in acts.items():
        for a in lst:
            if a["type"] == "FLIGHT" and a["rid"]:
                rev_flights[a["rid"]].append((k, a))
    failures = 0
    for rid, r in dec.items():
        n = len(rev_flights.get(rid, []))
        if r["decision"] == "ACCEPT":
            if n != 1:
                failures += 1
                bad("F1", f"{rid} accepted but has {n} revenue flights (service failure)")
        elif n:
            bad("F2", f"{rid} rejected but flown")
    for rid, fl in rev_flights.items():
        if rid not in reqs:
            bad("F2", f"flight for unknown request {rid}")
            continue
        q = reqs[rid]
        for k, a in fl:
            if a["frm"] != q["o"] or a["to"] != q["d"]:
                bad("F3", f"{rid} flown {a['frm']}->{a['to']}, booked {q['o']}->{q['d']}")
            if not (q["e"] <= a["s"] <= q["l"]):
                bad("F3", f"{rid} departs {a['s']} outside [{q['e']}, {q['l']}]")

    # per-aircraft checks --------------------------------------------------
    occupancy = {p: [0] * _DAY_MINUTES for p in _PORTS}
    min_landing = math.inf
    km_total = km_dead = kwh = revenue = 0.0
    served = set()
    for k in range(fleet):
        lst = sorted(acts.get(k, []), key=lambda a: (a["s"], a["type"] != "STINT"))
        here = _START_PORT_CYCLE[k % len(_START_PORT_CYCLE)]
        soc = soc0
        t_free = 0
        last_landing = None           # (time, request id) of last revenue landing
        for i, a in enumerate(lst):
            tag = f"A{k} {a['type']} [{a['s']},{a['e']})"
            if not (0 <= a["s"] < a["e"] <= _DAY_MINUTES):
                bad("T2", f"{tag} outside the day or empty")
            if a["s"] < t_free:
                bad("C2", f"{tag} overlaps previous activity ending {t_free}")
            if a["frm"] != here:
                bad("C2" if i else "C1", f"{tag} starts at {a['frm']} but aircraft is at {here}")
            if a["type"] == "STINT":
                if a["e"] - a["s"] < _TURNAROUND:
                    bad("T2", f"{tag} shorter than 5 min")
                soc = _after_charging(soc, a["e"] - a["s"], speed)
                if soc > 100.0 + _TOL:
                    bad("E2", f"{tag} SoC {soc:.4f} > 100")
                for t in range(max(0, a["s"]), min(_DAY_MINUTES, a["e"])):
                    occupancy[a["frm"]][t] += 1
            elif a["type"] == "FLIGHT":
                if a["to"] not in _PORTS or a["to"] == a["frm"]:
                    bad("C2", f"{tag} bad destination {a['to']}")
                    continue
                km = _km(a["frm"], a["to"])
                if a["e"] - a["s"] != _flight_minutes(km):
                    bad("T1", f"{tag} lasts {a['e'] - a['s']} min, expected {_flight_minutes(km)}")
                prev = lst[i - 1] if i > 0 else None
                nxt = lst[i + 1] if i + 1 < len(lst) else None
                if not (prev and prev["type"] == "STINT" and prev["e"] == a["s"]
                        and prev["frm"] == a["frm"]):
                    bad("C3", f"{tag} departure does not end a stint at {a['frm']}")
                if not (nxt and nxt["type"] == "STINT" and nxt["s"] == a["e"]
                        and nxt["frm"] == a["to"]):
                    bad("C3", f"{tag} landing does not start a stint at {a['to']}")
                soc -= _flight_energy(km)
                min_landing = min(min_landing, soc)
                if soc < _MIN_LANDING_SOC - _TOL:
                    bad("E1", f"{tag} lands with SoC {soc:.4f} < 20")
                km_total += km
                kwh += _flight_energy(km)
                if a["rid"]:
                    if a["rid"] in reqs:
                        revenue += reqs[a["rid"]]["rev"]
                        served.add(a["rid"])
                        # R1: everything after the previous revenue landing stint
                        # and before this departure was planned at reveal time
                        reveal = max(0, reqs[a["rid"]]["e"] - 30)
                        for b in lst[:i]:
                            if last_landing is not None and b["s"] <= last_landing:
                                continue
                            if b["s"] < reveal:
                                bad("R1", f"A{k} {b['type']} at {b['s']} prepares {a['rid']} "
                                          f"before its reveal time {reveal}")
                    last_landing = a["e"]
                else:
                    km_dead += km
                here = a["to"]
            else:
                bad("C2", f"{tag} unknown type")
            t_free = a["e"]

    # P1 ----------------------------------------------------------------------
    for p, row in occupancy.items():
        over = [t for t, n in enumerate(row) if n > pad_of[p]]
        if over:
            bad("P1", f"{p} over capacity ({max(row)} > {pad_of[p]}) at minutes "
                      f"{over[0]}..{over[-1]} ({len(over)} min)")

    profit = revenue - 1.5 * km_total - 0.3 * kwh
    kpis = dict(served=len(served), failures=failures, revenue=round(revenue, 2),
                km_total=round(km_total, 1), km_deadhead=round(km_dead, 1),
                kwh=round(kwh, 1), profit=round(profit, 2),
                min_landing_soc=round(min_landing, 2) if served else "")
    for p in sorted(_PORTS):
        kpis[f"pad_util_{p}"] = (round(sum(occupancy[p]) / (pad_of[p] * _DAY_MINUTES), 3)
                                 if pad_of[p] > 0 else "")

    # K1 ----------------------------------------------------------------------
    if kpi_row:
        for key in ("served", "revenue", "km_total", "km_deadhead", "kwh", "profit",
                    "min_landing_soc"):
            if key in kpi_row and str(kpi_row[key]) != "" and kpis[key] != "":
                if abs(float(kpi_row[key]) - float(kpis[key])) > 0.011:
                    bad("K1", f"reported {key}={kpi_row[key]} but recomputed {kpis[key]}")

    return dict(ok=not violations, n_violations=len(violations),
                violations=violations, kpis=kpis)


def audit_files(requests, schedule, decisions, config=None, kpis=None) -> dict:
    cfg = {}
    if isinstance(config, dict):
        cfg = config
    elif config:
        with open(config) as f:
            cfg = json.load(f)
    kpi_row = _read_csv(kpis)[0] if kpis else None
    return audit(_read_csv(requests), _read_csv(schedule), _read_csv(decisions), cfg, kpi_row)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Independent Meridian constraint checker")
    ap.add_argument("--requests", required=True)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--config", help="JSON file; default = scenario defaults")
    ap.add_argument("--kpis", help="KPI CSV (one row) to cross-check")
    a = ap.parse_args(argv)
    rep = audit_files(a.requests, a.schedule, a.decisions, a.config, a.kpis)
    print("PASS" if rep["ok"] else f"FAIL ({rep['n_violations']} violations)")
    for v in rep["violations"][:50]:
        print("  " + v)
    for k, v in rep["kpis"].items():
        print(f"  {k:>16}: {v}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
