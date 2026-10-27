"""Unit and integration tests.  Run:  python -m pytest -q tests/"""
import copy
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "baseline"))

import greedy_jit as G  # noqa: E402
from meridian import auditor  # noqa: E402
from meridian.env import (ALL_MODES, Config, MeridianEnv, generate_requests,  # noqa: E402
                          jit_action, load_requests, run_policy)
from meridian.metrics import requests_to_rows, schedule_rows  # noqa: E402
from meridian.physics import PadBook, block_time, charge, energy, min_stint  # noqa: E402

REQ_CSV = os.path.join(ROOT, "data", "requests.csv")


# ---------------------------------------------------------------- physics
@pytest.mark.parametrize("d,bt", [(30.0, 14), (19.0, 11), (37.2, 17), (11.4, 8),
                                  (18.0, 10), (18.1, 11), (0.3, 5)])
def test_block_time_rounding(d, bt):
    assert block_time(d) == bt
    assert block_time(d) == G.block_time(d)


def test_energy():
    assert energy(19.0) == pytest.approx(31.0)


def test_charging_curve():
    assert charge(20.0, 30) == pytest.approx(80.0)          # 2 %/min below 80
    assert charge(80.0, 3) == pytest.approx(82.0)           # 2/3 %/min above 80
    assert charge(70.0, 10) == pytest.approx(80.0 + 5 * 2 / 3)  # crosses 80 mid-stint
    assert charge(99.0, 30) == 100.0                        # cap
    assert charge(20.0, 10, mult=1.5) == pytest.approx(50.0)
    # a stint split in two charges exactly like one long stint (stint merging)
    for s in (20.0, 55.5, 79.0, 85.0):
        for a in range(0, 25, 3):
            assert charge(charge(s, a), 17) == pytest.approx(charge(s, a + 17))


def test_min_stint_edges():
    assert min_stint(90.0, 50.0) == 5                       # already enough -> 5 min floor
    assert min_stint(20.0, 100.1) is None                   # impossible target
    assert charge(20.0, min_stint(20.0, 69.2)) >= 69.2
    assert charge(20.0, min_stint(20.0, 69.2) - 1) < 69.2


def test_physics_matches_baseline():
    rng = random.Random(1)
    for _ in range(2000):
        s = rng.uniform(0, 100); t = rng.randint(0, 60); m = rng.choice([0.75, 1, 1.5])
        assert charge(s, t, m) == G.charge(s, t, m)
        tgt = rng.uniform(20, 101)
        assert min_stint(s, tgt, m) == G.min_stint(s, tgt, m)


def test_pad_capacity_edges():
    pb = PadBook((1, 1, 1, 1, 1))
    pb.book(3, 10, 20)
    assert not pb.free(3, 19, 21)
    assert pb.free(3, 20, 25)          # [a, b) intervals: end minute is free again
    assert pb.free(3, 5, 10)
    assert not pb.free(3, 355, 361)    # day end
    pb0 = PadBook((0, 1, 1, 1, 1))
    assert not pb0.free(0, 0, 5)       # zero-pad vertiport (outage)


# ---------------------------------------------------------------- generator / fidelity
def test_generator_matches_baseline():
    for seed in (0, 7, 1234, 2000):
        for dm, ps in ((1.0, 0.0), (1.6, -30.0)):
            a = generate_requests(seed, dm, ps)
            b = G.generate_requests(seed, dm, ps)
            assert [vars(x) for x in a] == [vars(x) for x in b]


def test_fixed_instance_has_70_requests():
    assert len(load_requests(REQ_CSV)) == 70


@pytest.mark.parametrize("cfg", [Config(), Config(fleet_size=6), Config(fleet_size=10),
                                 Config(charge_mult=0.75), Config(pads=(2, 2, 2, 1, 1))])
def test_env_jit_reproduces_baseline(cfg):
    env = MeridianEnv(cfg)
    gcfg = G.Config(fleet_size=cfg.fleet_size, pads=cfg.pads, charge_mult=cfg.charge_mult)
    for seed in range(3000, 3015):
        ref = G.run_day(G.generate_requests(seed), gcfg)
        mine = run_policy(env, jit_action, seed=seed)
        for k, v in ref.kpis.items():
            assert mine[k] == v, (seed, k)
        got = [(a.aircraft, a.type, a.frm, a.to, a.start, a.end, a.request_id)
               for a in env.merged_schedule()]
        exp = [(a.aircraft, a.type, a.frm, a.to, a.start, a.end, a.request_id)
               for a in ref.schedule]
        assert got == exp


# ---------------------------------------------------------------- determinism / masking
def _random_rollout(env, seed, action_seed, **kw):
    rng = random.Random(action_seed)
    obs = env.reset(seed=seed, **kw)
    acts = []
    while not env.done:
        legal = [a for a, ok in enumerate(obs["mask"]) if ok]
        a = rng.choice(legal)
        acts.append(a)
        obs, _, _, _ = env.step(a)
    return acts, [vars(a) for a in env.merged_schedule()], env.kpis()


def test_determinism_same_seed_same_actions():
    env1, env2 = MeridianEnv(Config()), MeridianEnv(Config())
    acts, s1, k1 = _random_rollout(env1, 42, 99)
    obs = env2.reset(seed=42)
    for a in acts:
        obs, _, _, _ = env2.step(a)
    assert [vars(a) for a in env2.merged_schedule()] == s1
    assert env2.kpis() == k1
    _, s3, _ = _random_rollout(MeridianEnv(Config()), 42, 99)
    assert s3 == s1


def test_masked_action_is_refused():
    env = MeridianEnv(Config())
    obs = env.reset(seed=5)
    while all(obs["mask"]):
        obs, _, _, _ = env.step(0)
    bad = obs["mask"].index(False)
    with pytest.raises(ValueError):
        env.step(bad)


def test_rewards_sum_to_profit():
    env = MeridianEnv(Config())
    rng = random.Random(3)
    obs = env.reset(seed=11)
    total = 0.0
    while not env.done:
        a = rng.choice([a for a, ok in enumerate(obs["mask"]) if ok])
        obs, r, _, _ = env.step(a)
        total += r
    assert total == pytest.approx(env.kpis()["profit"], abs=0.01)


def _audit_env(env, requests):
    return auditor.audit(
        [{k: str(v) for k, v in r.items()} for r in requests_to_rows(requests)],
        [{k: str(v) for k, v in r.items()} for r in schedule_rows(env)],
        [{k: str(v) for k, v in d.items()} for d in env.decisions],
        env.cfg.to_dict(), {k: str(v) for k, v in env.kpis().items()})


@pytest.mark.parametrize("cfg", [Config(), Config(fleet_size=10, charge_mult=0.75),
                                 Config(pads=(3, 1, 2, 1, 1))])
def test_any_unmasked_policy_is_safe(cfg):
    """Safety by construction: random choices among unmasked actions (all modes)
    always pass the independent auditor."""
    env = MeridianEnv(cfg, modes=ALL_MODES)
    for seed in range(20):
        _random_rollout(env, seed, seed + 1, demand_mult=1.3)
        rep = _audit_env(env, env.requests)
        assert rep["ok"], rep["violations"][:5]


# ---------------------------------------------------------------- auditor
def test_auditor_is_independent():
    imports = [ln.strip() for ln in open(auditor.__file__)
               if ln.strip().startswith(("import ", "from "))]
    allowed = {"argparse", "csv", "json", "math", "sys", "collections", "__future__"}
    for ln in imports:
        mod = ln.split()[1].split(".")[0]
        assert mod in allowed, ln


def _jit_logs():
    reqs = load_requests(REQ_CSV)
    env = MeridianEnv(Config())
    run_policy(env, jit_action, requests=reqs)
    R = [{k: str(v) for k, v in r.items()} for r in requests_to_rows(reqs)]
    S = [{k: str(v) for k, v in r.items()} for r in schedule_rows(env)]
    D = [{k: str(v) for k, v in d.items()} for d in env.decisions]
    return R, S, D, env.cfg.to_dict()


def test_auditor_passes_clean_jit():
    R, S, D, C = _jit_logs()
    rep = auditor.audit(R, S, D, C)
    assert rep["ok"], rep["violations"]
    assert rep["kpis"]["profit"] == pytest.approx(4011.64)


def _codes(rep):
    return {v.split(":")[0] for v in rep["violations"]}


def test_auditor_catches_pad_violation():
    R, S, D, C = _jit_logs()
    C = dict(C, pads=[3, 2, 2, 0, 1])          # V3 has no pad -> every V3 stint overflows
    assert "P1" in _codes(auditor.audit(R, S, D, C))
    R, S, D, C = _jit_logs()
    S = S + [dict(aircraft="A6", type="STINT", **{"from": "V3"}, to="", start="355",
                  end="360", request_id="")]
    S2 = [s for s in S]
    # put a second aircraft on V3's single pad at the same time as an existing stint
    v3 = next(s for s in S if s["from"] == "V3" and s["type"] == "STINT")
    other = "A0" if v3["aircraft"] != "A0" else "A1"
    S2.append(dict(aircraft=other, type="STINT", **{"from": "V3"}, to="",
                   start=v3["start"], end=v3["end"], request_id=""))
    assert "P1" in _codes(auditor.audit(R, S2, D, C))


def test_auditor_catches_reserve_violation():
    R, S, D, C = _jit_logs()
    C = dict(C, initial_soc=40.0)               # same plan, emptier batteries
    assert "E1" in _codes(auditor.audit(R, S, D, C))
    R, S, D, C = _jit_logs()
    C = dict(C, charge_mult=0.5)                # slower chargers than planned for
    assert "E1" in _codes(auditor.audit(R, S, D, C))


def test_auditor_catches_unserved_accepted_request():
    R, S, D, C = _jit_logs()
    rid = next(d["request_id"] for d in D if d["decision"] == "ACCEPT")
    S = [s for s in S if s["request_id"] != rid]
    rep = auditor.audit(R, S, D, C)
    assert "F1" in _codes(rep)
    assert rep["kpis"]["failures"] == 1


def test_auditor_catches_rejected_but_flown_and_window():
    R, S, D, C = _jit_logs()
    D2 = copy.deepcopy(D)
    acc = next(d for d in D2 if d["decision"] == "ACCEPT")
    acc["decision"] = "REJECT"
    assert "F2" in _codes(auditor.audit(R, S, D2, C))
    R2 = copy.deepcopy(R)
    r = next(r for r in R2 if r["id"] == acc["request_id"])
    r["earliest_dep"] = str(int(r["latest_dep"]) + 1)
    r["latest_dep"] = str(int(r["latest_dep"]) + 16)
    assert "F3" in _codes(auditor.audit(R2, S, D, C))


def test_auditor_catches_timing_and_continuity():
    R, S, D, C = _jit_logs()
    S2 = copy.deepcopy(S)
    f = next(s for s in S2 if s["type"] == "FLIGHT")
    f["end"] = str(int(f["end"]) - 1)           # too fast
    assert {"T1", "C3"} & _codes(auditor.audit(R, S2, D, C))
    S3 = copy.deepcopy(S)
    st = next(s for s in S3 if s["type"] == "STINT")
    st["start"] = str(int(st["end"]) - 3)       # 3-minute stint
    assert "T2" in _codes(auditor.audit(R, S3, D, C))
    S4 = copy.deepcopy(S)
    st = next(s for s in S4 if s["type"] == "STINT" and s["aircraft"] == "A0")
    st["from"] = "V4"                           # teleport
    assert {"C1", "C2", "C3"} & _codes(auditor.audit(R, S4, D, C))


def test_auditor_catches_action_before_reveal():
    R, S, D, C = _jit_logs()
    # take an aircraft's first stint and start it at t=0, before the request was known
    for s in S:
        if s["type"] == "STINT":
            rid_next = None
            break
    S2 = copy.deepcopy(S)
    first = {}
    for s in S2:
        first.setdefault(s["aircraft"], s)
    late = max(first.values(), key=lambda s: int(s["start"]))
    late["start"] = "0"
    rep = auditor.audit(R, S2, D, C)
    assert "R1" in _codes(rep)
