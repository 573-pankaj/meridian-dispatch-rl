#!/usr/bin/env python3
"""Side experiment (rejected design): Monte-Carlo rollout teacher.

Runs the rollout teacher of meridian/rollout.py on a few test days and
compares it with JIT and the precharge heuristic. Writes
results/rollout_teacher_check.csv. Slow (~12 s per day per M=4).

  python experiments/rollout_teacher_check.py --days 10 --m 4
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import precharge_heuristic  # noqa: E402
from meridian.env import JIT, Config, MeridianEnv, jit_action, run_policy  # noqa: E402
from meridian.rollout import rollout_policy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=int, default=10)
ap.add_argument("--m", type=int, default=4)
a = ap.parse_args()
rows = []
for s in range(2000, 2000 + a.days):
    j = run_policy(MeridianEnv(Config(), modes=(JIT,)), jit_action, seed=s)
    h = run_policy(MeridianEnv(Config()), precharge_heuristic, seed=s)
    t = time.perf_counter()
    r = run_policy(MeridianEnv(Config()), rollout_policy(m=a.m, seed=s), seed=s)
    rows.append(dict(seed=s, jit=j["profit"], precharge=h["profit"], rollout=r["profit"],
                     rollout_strategic_reject=r["strategic_reject"],
                     rollout_seconds=round(time.perf_counter() - t, 1)))
    print(rows[-1], flush=True)
os.makedirs("results", exist_ok=True)
with open("results/rollout_teacher_check.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
for k in ("jit", "precharge", "rollout"):
    print(k, round(sum(r[k] for r in rows) / len(rows), 1))
