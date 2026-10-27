#!/usr/bin/env python3
"""Train the dispatch policy: BC warm start from Greedy-JIT, then ES fine-tuning.

  python train.py                      # main policy  -> results/policy.npz
  python train.py --accept-bonus 50 --out results/policy_bonus.npz   # rejected reward variant

Seeds: training days use seeds 0..999 (demand_mult ~ U[0.8, 1.3]),
validation (model selection) uses seeds 5000..5029, the test set in
evaluate.py uses seeds 2000..2199. The three ranges never overlap.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from multiprocessing import Pool

import numpy as np

from meridian.agent import Adam, Policy, Scorer, candidate_features, centered_ranks
from meridian.env import Config, MeridianEnv, jit_action

TRAIN_SEEDS = range(0, 1000)
VAL_SEEDS = range(5000, 5030)

_ENV = None
_SCORER = None


def _init_worker(accept_bonus: float, hidden):
    global _ENV, _SCORER
    _ENV = MeridianEnv(Config(), accept_bonus=accept_bonus)
    _SCORER = Scorer(hidden=hidden)


def _episode_return(args):
    """Return (training return, true profit) of one day for parameters theta."""
    theta, seed, dmult = args
    pol = Policy(_SCORER, theta)
    obs = _ENV.reset(seed=seed, demand_mult=dmult)
    ret = 0.0
    while not _ENV.done:
        obs, r, _, _ = _ENV.step(pol(obs, _ENV))
        ret += r
    return ret, _ENV.kpis()["profit"]


# --------------------------------------------------------------------------
def behaviour_cloning(scorer: Scorer, n_days: int, rng: np.random.Generator,
                      epochs: int = 30, lr: float = 3e-3, log=print,
                      stop_acc: float = 0.999, target_margin: float = 1.0, l2: float = 1e-4):
    env = MeridianEnv(Config())
    data = []
    for i in range(n_days):
        seed = int(rng.choice(TRAIN_SEEDS))
        obs = env.reset(seed=seed, demand_mult=float(rng.uniform(0.8, 1.3)))
        while not env.done:
            a = jit_action(obs, env)
            X, actions = candidate_features(obs, env)
            if len(actions) > 1:
                data.append((X, actions.index(a)))
            obs, _, _, _ = env.step(a)
    log(f"BC: {len(data)} decision states from {n_days} JIT days")
    opt = Adam(scorer.n_params, lr=lr)
    for ep in range(epochs):
        order = rng.permutation(len(data))
        loss = acc = 0.0
        for j in order:
            X, y = data[j]
            s, hs = scorer.forward(X, cache=True)
            p = np.exp(s - s.max()); p /= p.sum()
            loss -= math.log(max(p[y], 1e-12))
            acc += float(np.argmax(s) == y)
            g = -p; g[y] += 1.0          # gradient of log p_y (ascent)
            scorer.theta += opt.step(scorer.backward(hs, g) - l2 * scorer.theta)
        log(f"BC epoch {ep:2d}  loss {loss / len(data):.4f}  acc {acc / len(data):.4f}")
        if acc / len(data) >= stop_acc:
            break
    # Temperature calibration. Cross-entropy pushes the score gap between the
    # JIT choice and the runner-up towards infinity; then small ES noise never
    # changes a decision and ES has no signal. Rescale the output layer so the
    # median gap is `target_margin` (arg-max decisions are unchanged).
    gaps = []
    for X, y in data:
        s = scorer.forward(X)
        gaps.append(s[y] - np.max(np.delete(s, y)))
    med = float(np.median(gaps))
    if med > 0:
        scale = target_margin / med
        (W, c) = scorer.unpack()[-1]
        n_last = W.size + c.size
        scorer.theta[-n_last:] *= scale
    log(f"BC calibration: median score gap {med:.3f} -> {target_margin}")
    return acc / len(data)



def evaluate_val(pool, theta) -> float:
    """Mean validation RETURN under the training reward. For the main reward this
    equals the true profit. For a reward variant, model selection must use that
    variant's own reward; selecting on true profit would hide its flaws."""
    res = pool.map(_episode_return, [(theta, s, 1.0) for s in VAL_SEEDS])
    return float(np.mean([r for r, _ in res]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bc-days", type=int, default=150)
    ap.add_argument("--bc-epochs", type=int, default=25)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--pairs", type=int, default=16, help="antithetic pairs per iteration")
    ap.add_argument("--batch-days", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--accept-bonus", type=float, default=0.0,
                    help="reward variant: +B per accepted request (rejected design)")
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="results/policy.npz")
    ap.add_argument("--time-limit", type=float, default=40 * 60, help="seconds (hard cap)")
    a = ap.parse_args(argv)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    stem = os.path.splitext(a.out)[0]
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    t_start = time.perf_counter()
    rng = np.random.default_rng(a.seed)
    hidden = ()  # linear scorer over candidate + interaction features
    scorer = Scorer(hidden=hidden, seed=a.seed)
    hw = dict(platform=platform.platform(), processor=platform.processor(),
              python=platform.python_version(), cpu_count=os.cpu_count(), jobs=a.jobs)
    log(f"hardware: {json.dumps(hw)}")
    log(f"params: {scorer.n_params}, features: {scorer.shapes[0][0]}")

    # ---- Stage 1 ---------------------------------------------------------
    bc_acc = behaviour_cloning(scorer, a.bc_days, rng, epochs=a.bc_epochs, log=log)
    t_bc = time.perf_counter() - t_start
    log(f"Stage 1 done in {t_bc:.1f}s")

    # ---- Stage 2: ES -----------------------------------------------------
    curve = []
    with Pool(a.jobs, initializer=_init_worker, initargs=(a.accept_bonus, hidden)) as pool:
        theta = scorer.theta.copy()
        best_theta, best_val = theta.copy(), evaluate_val(pool, theta)
        curve.append(dict(iter=0, time_s=round(time.perf_counter() - t_start, 1),
                          train_return=None, train_profit=None, val_return=round(best_val, 2)))
        log(f"iter   0  val return {best_val:.2f} (BC policy)")
        opt = Adam(theta.size, lr=a.lr)
        for it in range(1, a.iters + 1):
            if time.perf_counter() - t_start > a.time_limit:
                log("time limit reached, stopping")
                break
            seeds = [int(s) for s in rng.choice(TRAIN_SEEDS, a.batch_days, replace=False)]
            mults = [float(x) for x in rng.uniform(0.8, 1.3, a.batch_days)]
            eps = rng.standard_normal((a.pairs, theta.size))
            thetas = [theta + a.sigma * e for e in eps] + [theta - a.sigma * e for e in eps]
            jobs = [(th, s, m) for th in thetas for s, m in zip(seeds, mults)]
            res = pool.map(_episode_return, jobs, chunksize=4)
            R = np.array([r for r, _ in res]).reshape(len(thetas), a.batch_days).mean(1)
            P = np.array([p for _, p in res]).reshape(len(thetas), a.batch_days).mean(1)
            w = centered_ranks(R)
            wp, wm = w[:a.pairs], w[a.pairs:]
            grad = ((wp - wm)[:, None] * eps).sum(0) / (2 * a.pairs * a.sigma)
            theta = theta + opt.step(grad - 1e-4 * theta)   # small L2 decay
            row = dict(iter=it, time_s=round(time.perf_counter() - t_start, 1),
                       train_return=round(float(R.mean()), 2),
                       train_profit=round(float(P.mean()), 2), val_return=None)
            if it % a.val_every == 0 or it == a.iters:
                v = evaluate_val(pool, theta)
                row["val_return"] = round(v, 2)
                if v > best_val:
                    best_val, best_theta = v, theta.copy()
                log(f"iter {it:3d}  t={row['time_s']:6.1f}s  pop return {R.mean():8.2f}  "
                    f"pop profit {P.mean():8.2f}  val return {v:8.2f}  best {best_val:8.2f}")
            curve.append(row)

    total = time.perf_counter() - t_start
    scorer.theta = best_theta
    scorer.save(a.out, val_return=best_val, train_seconds=total, accept_bonus=a.accept_bonus)
    meta = dict(hardware=hw, args=vars(a), bc_accuracy=bc_acc, bc_seconds=round(t_bc, 1),
                total_train_seconds=round(total, 1), best_val_return=round(best_val, 2),
                n_params=int(scorer.n_params))
    with open(stem + "_train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    import csv
    with open(stem + "_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
        w.writeheader(); w.writerows(curve)
    with open(stem + "_train_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log(f"done: {total / 60:.1f} min total, best val profit {best_val:.2f}, saved {a.out}")


if __name__ == "__main__":
    main()
