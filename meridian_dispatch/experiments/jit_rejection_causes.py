#!/usr/bin/env python3
"""Why does Greedy-JIT reject? For each forced rejection on the test days,
relax one resource at a time and check if some aircraft could then serve it:
  pads : unlimited pads everywhere
  soc  : every aircraft at 100 % SoC (charging never binds)
  busy : every aircraft free now (no earlier commitment)
Writes results/jit_rejection_causes.md.
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meridian.env import JIT, Config, MeridianEnv, jit_action, plan_for_aircraft  # noqa: E402
from meridian.physics import PadBook  # noqa: E402


def main(days=200):
    single, combo = Counter(), Counter()
    n = 0
    big = PadBook((99,) * 5)
    for s in range(2000, 2000 + days):
        env = MeridianEnv(Config(), modes=(JIT,))
        obs = env.reset(seed=s)
        while not env.done:
            a = jit_action(obs, env)
            if a == 0:
                n += 1
                req, fleet = obs["request"], obs["fleet"]
                ok = dict(
                    pads=any(plan_for_aircraft(k, st, req, big, env.cfg) for k, st in enumerate(fleet)),
                    soc=any(plan_for_aircraft(k, (l, f, 100.0), req, obs["pads"], env.cfg)
                            for k, (l, f, _) in enumerate(fleet)),
                    busy=any(plan_for_aircraft(k, (l, 0, s_), req, obs["pads"], env.cfg)
                             for k, (l, _, s_) in enumerate(fleet)))
                for k, v in ok.items():
                    single[k] += v
                combo[tuple(k for k, v in ok.items() if v) or ("none of the three",)] += 1
            obs, _, _, _ = env.step(a)
    L = [f"# Causes of Greedy-JIT forced rejections ({days} test days, {n} rejections)\n",
         "| Relaxed resource | Rejections that become servable |", "|---|---|"]
    for k, label in (("soc", "battery (all aircraft at 100 % SoC)"),
                     ("pads", "pads (unlimited)"), ("busy", "aircraft time (no earlier commitment)")):
        L.append(f"| {label} | {single[k]} ({100 * single[k] / n:.0f}%) |")
    L += ["", "| Any single relaxation that works | Count |", "|---|---|"]
    for k, v in combo.most_common():
        L.append(f"| {' + '.join(k)} | {v} |")
    open("results/jit_rejection_causes.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
