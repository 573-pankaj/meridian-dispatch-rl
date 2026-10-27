"""KPIs, battery-cycle counting and paired statistics."""
from __future__ import annotations

import csv
import json
import os
import time
from typing import Callable, Dict, List, Sequence

import numpy as np
from scipy import stats

from .env import Activity, MeridianEnv
from .physics import VERTIPORTS


def battery_metrics(schedule: Sequence[Activity], fleet_size: int) -> dict:
    """Battery-use metrics. Definitions (stated in REPORT.md):

    efc_discharge = sum over flights of SoC used / 100   (equivalent full cycles)
    efc_charge    = sum over stints of SoC added / 100
    micro_cycles  = number of charge -> discharge reversals = number of flights
                    that follow a stint with net charging (> 0.01 %)
    deep_landings = landings with SoC < 30 % (deep discharge, ageing driver)
    """
    dis = chg = 0.0
    micro = deep = 0
    by_ac: Dict[int, List[Activity]] = {}
    for a in schedule:
        by_ac.setdefault(a.aircraft, []).append(a)
    for acts in by_ac.values():
        acts = sorted(acts, key=lambda a: (a.start, a.type != "STINT"))
        charged_since_flight = False
        for a in acts:
            if a.type == "FLIGHT":
                dis += a.soc_start - a.soc_end
                if charged_since_flight:
                    micro += 1
                charged_since_flight = False
                if a.soc_end < 30.0:
                    deep += 1
            else:
                gain = a.soc_end - a.soc_start
                chg += gain
                if gain > 0.01:
                    charged_since_flight = True
    return dict(efc_discharge=round(dis / 100, 3), efc_charge=round(chg / 100, 3),
                efc_per_aircraft=round(dis / 100 / max(1, fleet_size), 3),
                micro_cycles=micro, deep_landings=deep)


def write_episode(env: MeridianEnv, stem: str) -> Dict[str, str]:
    """Write schedule, decisions, KPIs and config for the auditor."""
    paths = dict(schedule=stem + "_schedule.csv", decisions=stem + "_decisions.csv",
                 kpis=stem + "_kpis.csv", config=stem + "_config.json")
    with open(paths["schedule"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aircraft", "type", "from", "to", "start", "end", "request_id",
                    "soc_start", "soc_end"])
        for a in env.merged_schedule():
            w.writerow([f"A{a.aircraft}", a.type, VERTIPORTS[a.frm],
                        VERTIPORTS[a.to] if a.to is not None else "", a.start, a.end,
                        a.request_id, round(a.soc_start, 2), round(a.soc_end, 2)])
    with open(paths["decisions"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(env.decisions[0].keys()))
        w.writeheader(); w.writerows(env.decisions)
    k = env.kpis()
    with open(paths["kpis"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(k.keys()))
        w.writeheader(); w.writerow(k)
    with open(paths["config"], "w") as f:
        json.dump(env.cfg.to_dict(), f)
    return paths


def requests_to_rows(requests) -> List[dict]:
    return [dict(id=r.id, origin=VERTIPORTS[r.origin], dest=VERTIPORTS[r.dest],
                 earliest_dep=r.earliest_dep, latest_dep=r.latest_dep,
                 dist_km=r.dist_km, revenue=int(r.revenue) if float(r.revenue).is_integer()
                 else r.revenue) for r in requests]


def schedule_rows(env: MeridianEnv) -> List[dict]:
    return [dict(aircraft=f"A{a.aircraft}", type=a.type, **{"from": VERTIPORTS[a.frm]},
                 to=VERTIPORTS[a.to] if a.to is not None else "", start=a.start, end=a.end,
                 request_id=a.request_id) for a in env.merged_schedule()]


def run_episode(env: MeridianEnv, policy: Callable, **reset_kwargs) -> dict:
    """One day with timing. Returns KPIs + battery metrics + ms per day."""
    t0 = time.perf_counter()
    obs = env.reset(**reset_kwargs)
    while not env.done:
        obs, _, _, _ = env.step(policy(obs, env))
    ms = (time.perf_counter() - t0) * 1000
    out = env.kpis()
    out.update(battery_metrics(env.schedule, env.cfg.fleet_size))
    out["ms_per_day"] = round(ms, 2)
    out["dh_per_served"] = round(out["km_deadhead"] / max(1, out["served"]), 3)
    return out


def paired_summary(a: Sequence[float], b: Sequence[float]) -> dict:
    """b - a, paired over the same days."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    out = dict(mean_a=a.mean(), sd_a=a.std(ddof=1) if len(a) > 1 else 0.0,
               mean_b=b.mean(), sd_b=b.std(ddof=1) if len(b) > 1 else 0.0,
               median_a=float(np.median(a)), median_b=float(np.median(b)),
               mean_diff=d.mean(), ci95=1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0,
               win_rate=float((d > 0).mean()), loss_rate=float((d < 0).mean()))
    if len(d) > 1 and np.any(d != 0):
        out["t_p"] = float(stats.ttest_rel(b, a).pvalue)
        out["wilcoxon_p"] = float(stats.wilcoxon(d[d != 0]).pvalue) if np.sum(d != 0) > 0 else 1.0
    else:
        out["t_p"] = out["wilcoxon_p"] = 1.0
    return out
