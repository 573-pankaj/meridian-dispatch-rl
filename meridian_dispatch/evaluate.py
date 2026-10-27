#!/usr/bin/env python3
"""Evaluate trained policies against Greedy-JIT and write every table.

  python evaluate.py                       # full run (test seeds 2000..2199)
  python evaluate.py --days 40 --quick     # smoke run

All episodes (every policy, every setting) are re-checked by the independent
auditor (meridian/auditor.py). The fixed 70-request instance is written to
results/fixed/ as schedule/decision/KPI CSVs, and audited from those files.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import time
from collections import defaultdict
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from meridian import auditor
from meridian.agent import FEATURE_NAMES, Policy, Scorer
from meridian.env import (ALL_MODES, JIT, PRECHARGE, Config, MeridianEnv, Request,
                          generate_requests, jit_action, load_requests)
from meridian.metrics import (paired_summary, requests_to_rows, run_episode,
                              schedule_rows, write_episode)

TEST_SEEDS = list(range(2000, 2200))
OUT = "results"


# --------------------------------------------------------------------------
# Policies (built inside each worker)
# --------------------------------------------------------------------------
def precharge_heuristic(obs, env):
    """Hand-written baseline: JIT selection rule, but prefer the PRECHARGE
    version of a plan when it exists (charges longer when a pad is idle)."""
    K = env.cfg.fleet_size
    pref = [env.modes.index(PRECHARGE), env.modes.index(JIT)]
    best, bk = 0, None
    for rank, mi in enumerate(pref):
        for k in range(K):
            p = obs["candidates"][mi * K + k]
            if p is None:
                continue
            key = (p.deadhead_km, p.departure, rank, k)
            if bk is None or key < bk:
                best, bk = 1 + mi * K + k, key
    return best


_POL = {}


def _init(paths):
    _POL.clear()
    _POL["JIT"] = ((JIT,), jit_action)
    _POL["JIT+PRE"] = (ALL_MODES, precharge_heuristic)
    for name, path in paths.items():
        if path and os.path.exists(path):
            _POL[name] = (ALL_MODES, Policy(Scorer.load(path)))


def _audit(env) -> dict:
    s = lambda d: {k: str(v) for k, v in d.items()}  # noqa: E731
    return auditor.audit([s(r) for r in requests_to_rows(env.requests)],
                         [s(r) for r in schedule_rows(env)],
                         [s(d) for d in env.decisions], env.cfg.to_dict(),
                         s(env.kpis()))


def _task(args):
    setting, cfg_d, policy, seed, gen, requests = args
    modes, pol = _POL[policy]
    env = MeridianEnv(Config(**cfg_d), modes=modes)
    if requests is not None:
        k = run_episode(env, pol, requests=requests)
    else:
        k = run_episode(env, pol, seed=seed, **gen)
    rep = _audit(env)
    k.update(setting=setting, policy=policy, seed=seed, audit_ok=rep["ok"],
             audit_violations=rep["n_violations"], audit_failures=rep["kpis"]["failures"],
             audit_profit=rep["kpis"]["profit"])
    return k


# --------------------------------------------------------------------------
def settings_list(quick: bool):
    base = dict(fleet_size=8, pads=(3, 2, 2, 1, 1), charge_mult=1.0, initial_soc=90.0)
    S = [("nominal", base, {})]
    for dm in (0.7, 0.9, 1.1, 1.3, 1.6, 2.0):
        S.append((f"demand x{dm}", base, dict(demand_mult=dm)))
    for ps in (-45, -30, -15, 15, 30, 60):
        S.append((f"peak {ps:+d} min", base, dict(peak_shift=float(ps))))
    for cm in (0.75, 0.9, 1.1, 1.5):
        S.append((f"charge x{cm}", dict(base, charge_mult=cm), {}))
    for pads in ((2, 2, 2, 1, 1), (3, 1, 2, 1, 1), (3, 2, 1, 1, 1), (3, 2, 2, 1, 0)):
        S.append((f"pads {','.join(map(str, pads))}", dict(base, pads=pads), {}))
    for fl in (6, 10):
        S.append((f"fleet {fl}", dict(base, fleet_size=fl), {}))
    if quick:
        S = S[:3]
    return S


def jitter_instance(reqs, rng, max_shift):
    """Perturb the fixed instance: shift each request's window by U{-s..s} min
    (clipped to [0, 299]), keep route and price. Small input noise test."""
    out = []
    for r in reqs:
        e = min(299, max(0, r.earliest_dep + rng.randint(-max_shift, max_shift)))
        out.append(Request(r.id, r.origin, r.dest, e, e + 15, r.dist_km, r.revenue))
    return out


def fmt(x, nd=1):
    return f"{x:,.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=f"{OUT}/policy.npz")
    ap.add_argument("--bonus-policy", default=f"{OUT}/policy_bonus.npz")
    ap.add_argument("--days", type=int, default=len(TEST_SEEDS))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(f"{OUT}/fixed", exist_ok=True)
    seeds = TEST_SEEDS[:a.days]
    paths = {"RL": a.policy, "RL-bonus": a.bonus_policy}
    _init(paths)
    pols = list(_POL)
    print("policies:", pols)
    t0 = time.perf_counter()

    # ---------------------------------------------------------------- fixed instance
    fixed = load_requests("data/requests.csv")
    fixed_rows = []
    for name in pols:
        modes, pol = _POL[name]
        env = MeridianEnv(Config(), modes=modes)
        k = run_episode(env, pol, requests=fixed)
        stem = f"{OUT}/fixed/{name.lower().replace('+', '_')}"
        p = write_episode(env, stem)
        rep = auditor.audit_files("data/requests.csv", p["schedule"], p["decisions"],
                                  p["config"], p["kpis"])
        with open(stem + "_audit.txt", "w") as f:
            f.write(("PASS" if rep["ok"] else "FAIL") + "\n" + "\n".join(rep["violations"]) + "\n")
        k.update(policy=name, audit=("PASS" if rep["ok"] else f"FAIL({rep['n_violations']})"))
        fixed_rows.append(k)
        print(f"fixed instance {name:9s} profit {k['profit']:9.2f} served {k['served']} "
              f"audit {k['audit']}")

    # jitter test on the fixed instance (input noise)
    jit_rows = []
    for shift in (2, 5, 10):
        rng = random.Random(shift)
        variants = [jitter_instance(fixed, rng, shift) for _ in range(30)]
        tasks = [(f"jitter ±{shift}", Config().to_dict(), p, i, None, v)
                 for p in pols for i, v in enumerate(variants)]
        with Pool(a.jobs, initializer=_init, initargs=(paths,)) as pool:
            jit_rows += pool.map(_task, tasks, chunksize=4)

    # ---------------------------------------------------------------- generated days
    rows = []
    S = settings_list(a.quick)
    tasks = []
    for (name, cfg_d, gen) in S:
        cfg_d = dict(cfg_d, pads=tuple(cfg_d["pads"]))
        for p in pols:
            for s in seeds:
                tasks.append((name, cfg_d, p, s, gen, None))
    with Pool(a.jobs, initializer=_init, initargs=(paths,)) as pool:
        rows = pool.map(_task, tasks, chunksize=8)
    elapsed = time.perf_counter() - t0
    print(f"{len(rows) + len(jit_rows)} episodes in {elapsed:.0f}s")

    all_rows = rows + jit_rows
    keys = sorted({k for r in all_rows for k in r})
    with open(f"{OUT}/episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(all_rows)

    # ---------------------------------------------------------------- aggregate
    by = defaultdict(list)
    for r in all_rows:
        by[(r["setting"], r["policy"])].append(r)

    def col(setting, pol, key):
        return [float(r[key]) for r in sorted(by[(setting, pol)], key=lambda r: r["seed"])]

    summary = dict(policies=pols, n_days=len(seeds), episodes=len(all_rows),
                   audit_failures=sum(1 for r in all_rows if not r["audit_ok"]),
                   service_failures=sum(int(r["audit_failures"]) for r in all_rows),
                   eval_seconds=round(elapsed, 1), settings={})
    settings_order = [s[0] for s in S] + ["jitter ±2", "jitter ±5", "jitter ±10"]
    for st in settings_order:
        d = {}
        for p in pols:
            rr = by[(st, p)]
            if not rr:
                continue
            m = lambda k: statistics.fmean(float(r[k]) for r in rr)  # noqa: E731
            d[p] = dict(profit=m("profit"), profit_sd=statistics.stdev(float(r["profit"]) for r in rr),
                        profit_median=float(np.median([float(r["profit"]) for r in rr])),
                        requests=m("requests"), served=m("served"),
                        forced_reject=m("forced_reject"), strategic_reject=m("strategic_reject"),
                        km_deadhead=m("km_deadhead"), dh_per_served=m("dh_per_served"),
                        efc_discharge=m("efc_discharge"), efc_charge=m("efc_charge"),
                        micro_cycles=m("micro_cycles"), deep_landings=m("deep_landings"),
                        min_landing_soc=min(float(r["min_landing_soc"]) for r in rr
                                            if r["min_landing_soc"] != ""),
                        ms_per_day=m("ms_per_day"),
                        ms_per_decision=statistics.fmean(float(r["ms_per_day"]) / max(1, int(r["requests"])) for r in rr),
                        failures=sum(int(r["audit_failures"]) for r in rr),
                        audit_pass=sum(1 for r in rr if r["audit_ok"]), n=len(rr),
                        **{f"pad_util_V{v}": m(f"pad_util_V{v}") for v in range(5)
                           if all(r[f"pad_util_V{v}"] != "" for r in rr)},
                        **{f"mode_{md.name}": m(f"mode_{md.name}") for md in ALL_MODES
                           if f"mode_{md.name}" in rr[0]})
            if p != "JIT":
                d[p]["vs_JIT"] = paired_summary(col(st, "JIT", "profit"), col(st, p, "profit"))
        summary["settings"][st] = d
    summary["fixed_instance"] = fixed_rows
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ---------------------------------------------------------------- failure cases
    fails = []
    if "RL" in pols:
        for st in settings_order:
            j, r_ = by[(st, "JIT")], by[(st, "RL")]
            jm = {x["seed"]: x for x in j}
            for x in r_:
                y = jm.get(x["seed"])
                if y is not None:
                    fails.append(dict(setting=st, seed=x["seed"],
                                      d_profit=round(float(x["profit"]) - float(y["profit"]), 2),
                                      d_served=int(x["served"]) - int(y["served"]),
                                      d_deadhead=round(float(x["km_deadhead"]) - float(y["km_deadhead"]), 1),
                                      rl_strategic_reject=x["strategic_reject"],
                                      rl_modes="/".join(str(x.get(f"mode_{m.name}", 0)) for m in ALL_MODES),
                                      requests=x["requests"]))
        fails.sort(key=lambda r: r["d_profit"])
        with open(f"{OUT}/rl_vs_jit_per_day.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fails[0].keys()))
            w.writeheader(); w.writerows(fails)

    write_tables(summary, settings_order, pols)
    make_plots(summary, settings_order, pols)
    print(f"audit: {summary['audit_failures']} failing episodes, "
          f"{summary['service_failures']} service failures over {summary['episodes']} episodes")


# --------------------------------------------------------------------------
def write_tables(summary, settings_order, pols):
    L = []
    nom = summary["settings"]["nominal"]
    L.append(f"## Held-out test days (seeds 2000..{2000 + summary['n_days'] - 1}), nominal scenario\n")
    hdr = "| Metric | " + " | ".join(pols) + " |"
    L.append(hdr); L.append("|" + "---|" * (len(pols) + 1))

    def row(label, key, nd=1, fn=None):
        vals = []
        for p in pols:
            v = nom[p].get(key, "")
            vals.append(fn(nom[p]) if fn else (fmt(v, nd) if v != "" else ""))
        L.append(f"| {label} | " + " | ".join(vals) + " |")

    row("Profit, mean ± SD ($)", "profit", fn=lambda d: f"{d['profit']:,.1f} ± {d['profit_sd']:,.1f}")
    row("Profit, median ($)", "profit_median")
    row("Profit vs JIT, mean diff [95% CI]", "", fn=lambda d: (
        f"{d['vs_JIT']['mean_diff']:+,.1f} [±{d['vs_JIT']['ci95']:.1f}]" if "vs_JIT" in d else "—"))
    row("Days better / worse than JIT", "", fn=lambda d: (
        f"{100 * d['vs_JIT']['win_rate']:.0f}% / {100 * d['vs_JIT']['loss_rate']:.0f}%" if "vs_JIT" in d else "—"))
    row("Paired t-test p / Wilcoxon p", "", fn=lambda d: (
        f"{d['vs_JIT']['t_p']:.1e} / {d['vs_JIT']['wilcoxon_p']:.1e}" if "vs_JIT" in d else "—"))
    row("Requests per day", "requests")
    row("Served per day", "served", 2)
    row("Rejected: forced (no feasible plan)", "forced_reject", 2)
    row("Rejected: strategic (feasible, declined)", "strategic_reject", 2)
    row("Service failures (auditor), total", "failures", 0)
    row("Deadhead km per day", "km_deadhead")
    row("Deadhead km per served request", "dh_per_served", 2)
    row("Battery EFC, discharge (cycles/day, fleet)", "efc_discharge", 2)
    row("Battery EFC, charge (cycles/day, fleet)", "efc_charge", 2)
    row("Charge→discharge micro-cycles per day", "micro_cycles", 1)
    row("Landings below 30% SoC per day", "deep_landings", 2)
    row("Min landing SoC over all days (%)", "min_landing_soc", 2)
    for v in range(5):
        row(f"Pad utilisation V{v}", f"pad_util_V{v}", 3)
    row("Inference time per day (ms)", "ms_per_day", 1)
    row("Inference time per decision (ms)", "ms_per_decision", 2)
    row("Auditor pass", "", fn=lambda d: f"{d['audit_pass']}/{d['n']}")
    L.append("")

    L.append("## Fixed 70-request instance (data/requests.csv)\n")
    L.append("| Policy | Profit ($) | Served | Forced rej. | Strategic rej. | Deadhead km | "
             "EFC (dis.) | Min landing SoC | ms | Auditor |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k in summary["fixed_instance"]:
        L.append(f"| {k['policy']} | {k['profit']:,.2f} | {k['served']} | {k['forced_reject']} | "
                 f"{k['strategic_reject']} | {k['km_deadhead']} | {k['efc_discharge']} | "
                 f"{k['min_landing_soc']} | {k['ms_per_day']} | {k['audit']} |")
    L.append("")

    L.append("## Sensitivity (same trained weights, no retraining)\n")
    others = [p for p in pols if p != "JIT"]
    L.append("| Setting | JIT profit | " + " | ".join(f"{p} profit" for p in others)
             + " | " + " | ".join(f"{p} − JIT [95% CI]" for p in others)
             + " | JIT served | " + " | ".join(f"{p} served" for p in others)
             + " | JIT dh km | " + " | ".join(f"{p} dh km" for p in others) + " |")
    L.append("|" + "---|" * (1 + 1 + 2 * len(others) + 1 + len(others) + 1 + len(others)))
    for st in settings_order:
        d = summary["settings"].get(st)
        if not d or "JIT" not in d:
            continue
        cells = [st, fmt(d["JIT"]["profit"])]
        cells += [fmt(d[p]["profit"]) for p in others]
        cells += [f"{d[p]['vs_JIT']['mean_diff']:+,.1f} [±{d[p]['vs_JIT']['ci95']:.1f}]" for p in others]
        cells += [fmt(d["JIT"]["served"])] + [fmt(d[p]["served"]) for p in others]
        cells += [fmt(d["JIT"]["km_deadhead"])] + [fmt(d[p]["km_deadhead"]) for p in others]
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append(f"Episodes audited: {summary['episodes']}; failing audits: {summary['audit_failures']}; "
             f"service failures: {summary['service_failures']}.\n")
    with open(f"{OUT}/tables.md", "w") as f:
        f.write("\n".join(L))


def make_plots(summary, settings_order, pols):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    others = [p for p in pols if p != "JIT"]
    sts = [s for s in settings_order if s in summary["settings"] and "JIT" in summary["settings"][s]]
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(sts) + 1.5))
    y = np.arange(len(sts))
    h = 0.8 / max(1, len(others))
    for i, p in enumerate(others):
        diffs = [summary["settings"][s][p]["vs_JIT"]["mean_diff"] for s in sts]
        cis = [summary["settings"][s][p]["vs_JIT"]["ci95"] for s in sts]
        ax.barh(y + i * h, diffs, h, xerr=cis, label=p, capsize=2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y + h * (len(others) - 1) / 2); ax.set_yticklabels(sts)
    ax.invert_yaxis()
    ax.set_xlabel("Mean daily profit minus Greedy-JIT ($), 95% CI")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/sensitivity.png", dpi=130)
    plt.close(fig)

    for stem in ("policy", "policy_bonus"):
        path = f"{OUT}/{stem}_curve.csv"
        if not os.path.exists(path):
            continue
        rows = list(csv.DictReader(open(path)))
        it = [int(r["iter"]) for r in rows if r["val_return"]]
        vp = [float(r["val_return"]) for r in rows if r["val_return"]]
        it2 = [int(r["iter"]) for r in rows if r["train_profit"]]
        tp = [float(r["train_profit"]) for r in rows if r["train_profit"]]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(it2, tp, color="0.7", lw=0.8, label="ES population, training days (profit)")
        ax.plot(it, vp, "o-", ms=3, label="current weights, validation days (training reward)")
        if stem == "policy":
            val_seeds = range(5000, 5030)
            env = MeridianEnv(Config(), modes=(JIT,))
            jv = statistics.fmean(run_episode(env, jit_action, seed=s)["profit"] for s in val_seeds)
            env = MeridianEnv(Config())
            pv = statistics.fmean(run_episode(env, precharge_heuristic, seed=s)["profit"]
                                  for s in val_seeds)
            ax.axhline(jv, color="k", ls="--", lw=1, label=f"Greedy-JIT on validation days ({jv:,.0f})")
            ax.axhline(pv, color="C2", ls=":", lw=1.2, label=f"JIT+PRE heuristic ({pv:,.0f})")
        ax.set_xlabel("ES iteration"); ax.set_ylabel("Mean daily return ($)")
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/{stem}_learning_curve.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
