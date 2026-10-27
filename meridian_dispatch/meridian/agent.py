"""Policy: a shared per-candidate scoring network with hard action masking.

Design
------
* At each decision the env offers up to 1 + |modes| * K candidates (REJECT +
  aircraft x charging mode). Each candidate gets a feature vector phi(s, c).
* One small MLP scores every candidate with the SAME weights. The policy takes
  the arg-max over UNMASKED candidates only. Masked (infeasible) candidates are
  never scored, so the policy cannot choose them, whatever its weights are.
* Because weights are shared across candidates, the network does not depend on
  the fleet size: the same weights run with 6, 8 or 10 aircraft.

Training (see train.py)
-----------------------
Stage 1: behaviour cloning of Greedy-JIT (softmax cross-entropy over the
         unmasked candidates). This puts the policy at the baseline.
Stage 2: evolution strategies (ES, Salimans et al. 2017) on the true daily
         profit, with antithetic sampling and common random numbers.

Only numpy is needed.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .env import MeridianEnv, demand_rate
from .physics import DAY_END, ORIGIN_WEIGHTS

N_VERT = 5


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
FEATURE_NAMES: List[str] = (
    ["is_reject"] + [f"mode_{i}" for i in range(4)]
    + ["deadhead_km", "plan_profit", "dep_delay", "soc_dep", "soc_end", "min_land_soc",
       "busy_after", "post_stint", "first_stint", "pre_soc",
       "left_behind", "already_at_dest", "dest_pad_util_after", "origin_pad_util"]
    + ["time", "demand_ahead", "dist", "margin", "n_feasible", "idle_frac", "fleet_soc",
       "dest_weight", "dest_pad_scarcity", "origin_pad_scarcity"]
    + [f"orig_{v}" for v in range(N_VERT)] + [f"dest_{v}" for v in range(N_VERT)]
)
N_BASE = len(FEATURE_NAMES)
# Mode interactions: the same plan feature can mean different things for
# different charging modes (a long post-landing stint is fine at V0, which has
# 3 pads, and harmful at V3, which has 1). A linear scorer needs these products.
_INTER_PLAN = list(range(FEATURE_NAMES.index("deadhead_km"),
                         FEATURE_NAMES.index("origin_pad_util") + 1))  # plan-specific
_INTER_CTX = [FEATURE_NAMES.index(n) for n in
              ("time", "demand_ahead", "dest_pad_scarcity", "origin_pad_scarcity")]
for _m in range(4):
    FEATURE_NAMES += [f"m{_m}*{FEATURE_NAMES[j]}" for j in _INTER_PLAN + _INTER_CTX]
N_FEAT = len(FEATURE_NAMES)
_P0 = FEATURE_NAMES.index("deadhead_km")
LAM_MAX = 0.2 * 2.4


def candidate_features(obs: dict, env: MeridianEnv) -> Tuple[np.ndarray, List[int]]:
    """Return (X, actions): one row per unmasked action (row 0 is REJECT)."""
    req = obs["request"]
    fleet = obs["fleet"]
    pads = obs["pads"]
    cands = obs["candidates"]
    mask = obs["mask"]
    cfg = env.cfg
    K = cfg.fleet_size
    t0 = req.reveal
    o, d = req.origin, req.dest

    actions = [a for a in range(len(mask)) if mask[a]]
    X = np.zeros((len(actions), N_FEAT))
    n_int = len(_INTER_PLAN) + len(_INTER_CTX)

    feasible_ac = {cands[a - 1].aircraft for a in actions if a > 0}
    idle = sum(1 for (_, ff, _) in fleet if ff <= t0)
    ctx = np.zeros(10 + 2 * N_VERT)
    ctx[0] = t0 / DAY_END
    ctx[1] = demand_rate(t0 + 30) / LAM_MAX
    ctx[2] = req.dist_km / 40.0
    ctx[3] = req.direct_margin / 150.0
    ctx[4] = len(feasible_ac) / K
    ctx[5] = idle / K
    ctx[6] = sum(s for (_, _, s) in fleet) / (100.0 * K)
    ctx[7] = ORIGIN_WEIGHTS[d] / 1.8
    ctx[8] = 1.0 / max(1, cfg.pads[d])
    ctx[9] = 1.0 / max(1, cfg.pads[o])
    ctx[10 + o] = 1.0
    ctx[10 + N_VERT + d] = 1.0
    c0 = N_BASE - len(ctx)

    origin_util = pads.utilisation(o, t0, t0 + 45)
    loc_count = [0] * N_VERT
    for (loc, ff, _) in fleet:
        if ff <= t0 + 30:
            loc_count[loc] += 1

    # Context features go on the REJECT row only. Scores are compared within one
    # decision, so a feature added to every row has no effect; on the REJECT row
    # it makes the reject threshold depend on context (time, demand, fleet SoC,
    # route). Context still reaches accept rows through the mode interactions.
    ctx_int = ctx[[j - c0 for j in _INTER_CTX]]
    for r, a in enumerate(actions):
        if a == 0:
            X[r, 0] = 1.0
            X[r, c0:N_BASE] = ctx
            continue
        p = cands[a - 1]
        k = p.aircraft
        loc, ff, soc0 = fleet[k]
        X[r, 1 + p.mode] = 1.0
        here = loc_count[loc] - (1 if ff <= t0 + 30 else 0)
        X[r, _P0:_P0 + 14] = (
            p.deadhead_km / 40.0,                                   # deadhead_km
            p.profit / 150.0,                                       # plan_profit
            (p.departure - req.earliest_dep) / 15.0,                # dep_delay
            p.soc_at_departure / 100.0,                             # soc_dep
            p.new_state[2] / 100.0,                                 # soc_end
            p.min_landing_soc / 100.0,                              # min_land_soc
            (p.new_state[1] - t0) / 120.0,                          # busy_after
            (p.activities[-1].end - p.activities[-1].start) / 30.0,  # post_stint
            (p.activities[0].end - p.activities[0].start) / 30.0,    # first_stint
            soc0 / 100.0,                                           # pre_soc
            1.0 if here <= 0 else 0.0,                              # left_behind
            loc_count[d] / K,                                       # already_at_dest
            pads.utilisation(d, p.arrival, p.arrival + 60),         # dest_pad_util_after
            origin_util)                                            # origin_pad_util
        b = N_BASE + p.mode * n_int
        X[r, b:b + n_int - len(ctx_int)] = X[r, _INTER_PLAN]
        X[r, b + n_int - len(ctx_int):b + n_int] = ctx_int
    return X, actions


# --------------------------------------------------------------------------
# MLP scorer (numpy)
# --------------------------------------------------------------------------
class Scorer:
    def __init__(self, n_in: int = N_FEAT, hidden: Sequence[int] = (), seed: int = 0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, 1]
        self.shapes = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        params = []
        for (a, b) in self.shapes:
            params.append(rng.normal(0, 0.1 / math.sqrt(a), size=(a, b)))
            params.append(np.zeros(b))
        self.theta = np.concatenate([p.ravel() for p in params])

    @property
    def n_params(self) -> int:
        return self.theta.size

    def unpack(self, theta: Optional[np.ndarray] = None):
        theta = self.theta if theta is None else theta
        out, i = [], 0
        for (a, b) in self.shapes:
            W = theta[i:i + a * b].reshape(a, b); i += a * b
            c = theta[i:i + b]; i += b
            out.append((W, c))
        return out

    def forward(self, X: np.ndarray, theta: Optional[np.ndarray] = None,
                cache: bool = False):
        layers = self.unpack(theta)
        h = X
        hs = [h]
        for j, (W, c) in enumerate(layers):
            z = h @ W + c
            h = np.tanh(z) if j < len(layers) - 1 else z
            hs.append(h)
        s = h[:, 0]
        return (s, hs) if cache else s

    def backward(self, hs, grad_s: np.ndarray) -> np.ndarray:
        """Gradient of sum(grad_s * scores) w.r.t. theta."""
        layers = self.unpack()
        grads = []
        g = grad_s[:, None]
        for j in range(len(layers) - 1, -1, -1):
            W, c = layers[j]
            h_in = hs[j]
            gW = h_in.T @ g
            gc = g.sum(0)
            grads.append((gW, gc))
            if j > 0:
                g = (g @ W.T) * (1.0 - hs[j] ** 2)
        grads.reverse()
        return np.concatenate([np.concatenate([gW.ravel(), gc]) for gW, gc in grads])

    def save(self, path: str, **meta) -> None:
        np.savez(path, theta=self.theta, shapes=np.array(self.shapes),
                 feature_names=np.array(FEATURE_NAMES), **meta)

    @classmethod
    def load(cls, path: str) -> "Scorer":
        z = np.load(path, allow_pickle=False)
        shapes = [tuple(int(v) for v in s) for s in z["shapes"]]
        sc = cls(n_in=shapes[0][0], hidden=[s[1] for s in shapes[:-1]])
        sc.theta = z["theta"].astype(float)
        return sc


class Policy:
    """Greedy arg-max over unmasked candidates. Callable as policy(obs, env)."""

    def __init__(self, scorer: Scorer, theta: Optional[np.ndarray] = None):
        self.scorer = scorer
        self.theta = scorer.theta if theta is None else theta

    def __call__(self, obs: dict, env: MeridianEnv) -> int:
        X, actions = candidate_features(obs, env)
        if len(actions) == 1:
            return actions[0]
        s = self.scorer.forward(X, self.theta)
        return actions[int(np.argmax(s))]


# --------------------------------------------------------------------------
# Adam
# --------------------------------------------------------------------------
class Adam:
    def __init__(self, n: int, lr: float = 1e-2, b1: float = 0.9, b2: float = 0.999):
        self.m = np.zeros(n); self.v = np.zeros(n); self.t = 0
        self.lr, self.b1, self.b2 = lr, b1, b2

    def step(self, grad: np.ndarray) -> np.ndarray:
        """Returns the update to ADD for gradient ascent on `grad`."""
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return self.lr * mh / (np.sqrt(vh) + 1e-8)


def centered_ranks(x: np.ndarray) -> np.ndarray:
    """Rank shaping in [-0.5, 0.5]. Ties get the same (average) rank, so a
    population with identical returns gives a zero ES gradient, not noise."""
    from scipy.stats import rankdata
    r = rankdata(x) - 1.0
    return r / (len(x) - 1) - 0.5
