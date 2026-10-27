"""Monte-Carlo rollout teacher (one step of approximate policy iteration).

At a decision state, every candidate action is scored by
    immediate profit + mean profit-to-go of a fast base policy,
where the profit-to-go is simulated on M *sampled* futures. Futures are drawn
from the demand generator and keep only requests with earliest_dep later than
the current one, so the teacher never sees the real future requests (the
arrival process is Poisson, so this is the right conditional distribution under
nominal demand). All candidates share the same futures (common random numbers).

The teacher is slow (seconds per day). It is only used offline to label
training states; the deployed policy is the distilled scorer in agent.py.
"""
from __future__ import annotations

import random
from typing import Callable, List, Optional, Sequence, Tuple

from .env import (JIT, PRECHARGE, Config, MeridianEnv, Plan, Request,
                  generate_requests, plan_for_aircraft)
from .physics import PadBook

BASE_MODES = (PRECHARGE, JIT)


def base_choice(fleet, pads: PadBook, req: Request, cfg: Config) -> Optional[Plan]:
    """Base policy for rollouts: JIT selection rule, PRECHARGE version preferred."""
    best, bk = None, None
    for rank, mode in enumerate(BASE_MODES):
        for k in range(cfg.fleet_size):
            p = plan_for_aircraft(k, fleet[k], req, pads, cfg, mode, rank)
            if p is None:
                continue
            key = (p.deadhead_km, p.departure, rank, k)
            if bk is None or key < bk:
                best, bk = p, key
    return best


def simulate(fleet, pads: PadBook, cfg: Config, future: Sequence[Request]) -> float:
    """Profit of the base policy on `future`, from (fleet, pads). Mutates its inputs."""
    total = 0.0
    for req in future:
        p = base_choice(fleet, pads, req, cfg)
        if p is None:
            continue
        for a in p.activities:
            if a.type == "STINT":
                pads.book(a.frm, a.start, a.end)
        fleet[p.aircraft] = p.new_state
        total += p.profit
    return total


def sample_futures(after_ed: int, after_id: str, m: int, rng: random.Random,
                   demand_mult: float = 1.0) -> List[List[Request]]:
    out = []
    for _ in range(m):
        day = generate_requests(rng.randrange(10 ** 9), demand_mult)
        out.append([r for r in day if r.earliest_dep > after_ed])
    return out


def teacher_values(obs: dict, env: MeridianEnv, futures, max_cands: int = 8
                   ) -> List[Tuple[int, float]]:
    """(action, estimated value) for REJECT and up to `max_cands` candidates."""
    cands = obs["candidates"]
    acts = [a for a in range(1, len(obs["mask"])) if obs["mask"][a]]
    acts.sort(key=lambda a: (cands[a - 1].deadhead_km, cands[a - 1].departure,
                             cands[a - 1].mode, cands[a - 1].aircraft))
    acts = [0] + acts[:max_cands]
    out = []
    for a in acts:
        if a == 0:
            fleet0, pads0, imm = list(obs["fleet"]), obs["pads"], 0.0
        else:
            p = cands[a - 1]
            fleet0 = list(obs["fleet"]); fleet0[p.aircraft] = p.new_state
            pads0 = obs["pads"].copy()
            for x in p.activities:
                if x.type == "STINT":
                    pads0.book(x.frm, x.start, x.end)
            imm = p.profit
        v = 0.0
        for fut in futures:
            v += simulate(list(fleet0), pads0.copy(), env.cfg, fut)
        out.append((a, imm + v / len(futures)))
    return out


def rollout_policy(m: int = 4, seed: int = 0, demand_mult: float = 1.0, max_cands: int = 8):
    rng = random.Random(seed)

    def pol(obs, env):
        req = obs["request"]
        futs = sample_futures(req.earliest_dep, req.id, m, rng, demand_mult)
        vals = teacher_values(obs, env, futs, max_cands)
        return max(vals, key=lambda x: x[1])[0]
    return pol
