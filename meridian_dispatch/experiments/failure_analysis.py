#!/usr/bin/env python3
"""Failure-case analysis of the learned policy against Greedy-JIT.

  python experiments/failure_analysis.py      -> results/failure_analysis.md

1. What the policy does differently (modes, strategic rejections) and where.
2. The worst test days for RL vs JIT, request by request.
3. The fixed 70-request instance, request by request.
"""
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meridian.agent import Policy, Scorer  # noqa: E402
from meridian.env import (ALL_MODES, JIT, Config, MeridianEnv, jit_action,  # noqa: E402
                          load_requests)
from meridian.physics import VERTIPORTS  # noqa: E402

V = VERTIPORTS


def run(env, pol, **kw):
    obs = env.reset(**kw)
    log = []
    while not env.done:
        a = pol(obs, env)
        req = obs["request"]
        p = None if a == 0 else obs["candidates"][a - 1]
        best_dh = min((c.deadhead_km for c in obs["candidates"] if c is not None), default=None)
        log.append(dict(req=req, action=a, plan=p, feasible=any(obs["mask"][1:]),
                        best_dh=best_dh,
                        best_profit=max((c.profit for c in obs["candidates"] if c is not None),
                                        default=None)))
        obs, _, _, _ = env.step(a)
    return env.kpis(), log


def describe_day(title, kw, rl, out):
    kj, lj = run(MeridianEnv(Config(), modes=(JIT,)), jit_action, **kw)
    kr, lr = run(MeridianEnv(Config()), rl, **kw)
    out.append(f"### {title}\n")
    out.append(f"JIT profit {kj['profit']:.2f} ({kj['served']} served, {kj['km_deadhead']} dh km); "
               f"RL profit {kr['profit']:.2f} ({kr['served']} served, {kr['km_deadhead']} dh km, "
               f"{kr['strategic_reject']} strategic rejections). "
               f"RL modes JIT/TOPUP/PRECHARGE/LATE = "
               f"{kr['mode_JIT']}/{kr['mode_TOPUP']}/{kr['mode_PRECHARGE']}/{kr['mode_LATE']}.\n")
    sj = {x["req"].id for x in lj if x["action"] != 0}
    sr = {x["req"].id for x in lr if x["action"] != 0}
    only_j = [x for x in lr if x["req"].id in sj - sr]
    only_r = [x for x in lr if x["req"].id in sr - sj]
    gain = sum(x["plan"].profit for x in only_r)
    lost = sum(x["best_profit"] or x["req"].direct_margin for x in only_j)
    out.append(f"- Served by JIT only: {len(only_j)} (RL rejected "
               f"{sum(1 for x in only_j if x['feasible'])} of them by choice, "
               f"{sum(1 for x in only_j if not x['feasible'])} because no plan was feasible any more).")
    out.append(f"- Served by RL only: {len(only_r)}, with plan profit {gain:.0f}.")
    rows = [x for x in only_j if x["feasible"]][:8]
    if rows:
        out.append("\n| Request | Route | Window | Best plan profit | Min deadhead km |")
        out.append("|---|---|---|---|---|")
        for x in rows:
            r = x["req"]
            out.append(f"| {r.id} | {V[r.origin]}→{V[r.dest]} | {r.earliest_dep}–{r.latest_dep} | "
                       f"{x['best_profit']:.1f} | {x['best_dh']} |")
    out.append("")
    return kj, kr


def main():
    rl = Policy(Scorer.load("results/policy.npz"))
    out = ["# Failure analysis: learned policy vs Greedy-JIT\n"]

    # ---- behaviour profile over the 200 test days -------------------------
    strat_by_dest, strat_by_hour, mode_by_dest = Counter(), Counter(), defaultdict(Counter)
    strat_margin, strat_dh0 = [], 0
    n_strat = n_acc = 0
    per_day = []
    for s in range(2000, 2200):
        env = MeridianEnv(Config())
        k, log = run(env, rl, seed=s)
        kj = run(MeridianEnv(Config(), modes=(JIT,)), jit_action, seed=s)[0]
        per_day.append((k["profit"] - kj["profit"], s))
        for x in log:
            r = x["req"]
            if x["action"] == 0 and x["feasible"]:
                n_strat += 1
                strat_by_dest[V[r.dest]] += 1
                strat_by_hour[r.earliest_dep // 60] += 1
                strat_margin.append(x["best_profit"])
                strat_dh0 += x["best_dh"] == 0
            elif x["action"] != 0:
                n_acc += 1
                mode_by_dest[V[r.dest]][ALL_MODES[x["plan"].mode].name] += 1
    out.append("## 1. How the learned policy behaves (200 test days, nominal)\n")
    out.append(f"- Accepted {n_acc}, strategic rejections {n_strat} "
               f"({n_strat / 200:.2f} per day). In {strat_dh0} of the strategic rejections "
               f"a plan with zero deadhead existed.")
    if strat_margin:
        sm = sorted(strat_margin)
        out.append(f"- Profit of the best plan the policy turned down: median "
                   f"{sm[len(sm) // 2]:.1f}, min {sm[0]:.1f}, max {sm[-1]:.1f} $.")
    out.append("- Strategic rejections by destination: "
               + ", ".join(f"{d}: {strat_by_dest[d]}" for d in V))
    out.append("- Strategic rejections by departure hour (0 = 06:00): "
               + ", ".join(f"{h}: {strat_by_hour[h]}" for h in sorted(strat_by_hour)))
    out.append("\n| Destination (pads) | JIT mode | TOPUP | PRECHARGE | LATE |")
    out.append("|---|---|---|---|---|")
    for i, d in enumerate(V):
        c = mode_by_dest[d]
        tot = max(1, sum(c.values()))
        out.append(f"| {d} ({Config().pads[i]}) | " + " | ".join(
            f"{100 * c[m.name] / tot:.0f}%" for m in ALL_MODES) + " |")
    per_day.sort()
    out.append(f"\n- RL loses to JIT on {sum(1 for d, _ in per_day if d < 0)} of 200 days; "
               f"worst day {per_day[0][1]} ({per_day[0][0]:+.1f} $), best day "
               f"{per_day[-1][1]} ({per_day[-1][0]:+.1f} $).\n")

    out.append("## 2. Worst test days for RL\n")
    for d, s in per_day[:3]:
        describe_day(f"Seed {s} (RL − JIT = {d:+.1f} $)", dict(seed=s), rl, out)

    out.append("## 3. Fixed 70-request instance\n")
    describe_day("data/requests.csv", dict(requests=load_requests("data/requests.csv")), rl, out)

    with open("results/failure_analysis.md", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
