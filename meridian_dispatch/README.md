# Meridian eVTOL dispatch: simulator, independent checker and learned policy

Python 3.9+ (tested on 3.13). Dependencies: `numpy`, `scipy`, `matplotlib` (plots only),
`pytest` (tests only). No GPU and no deep-learning framework are needed.

```
pip install numpy scipy matplotlib pytest
bash run_all.sh                            # everything (~55 min); SKIP_MAIN_TRAINING=1 reuses results/policy.npz
# or step by step:
python -m pytest -q tests/                 # 33 tests, ~10 s
python train.py                            # main policy  -> results/policy.npz      (~10 min, 2 cores)
python train.py --accept-bonus 50 --out results/policy_bonus.npz   # rejected reward variant
python evaluate.py                         # all tables -> results/tables.md, summary.json, plots
python -m meridian.auditor --requests data/requests.csv \
    --schedule results/fixed/rl_schedule.csv --decisions results/fixed/rl_decisions.csv \
    --config results/fixed/rl_config.json --kpis results/fixed/rl_kpis.csv
```

The short note with all results is **[REPORT.md](REPORT.md)**.

## Layout

| Path | What it is |
|---|---|
| `baseline/greedy_jit.py` | The provided Greedy-JIT reference, unchanged |
| `meridian/physics.py` | Network, block time, energy, charging curve, pad book |
| `meridian/env.py` | Event-driven environment, candidate-plan generator (4 charging modes), action mask, demand generator, JIT policy |
| `meridian/auditor.py` | Independent checker. Imports only the standard library; reads CSV logs only |
| `meridian/agent.py` | Candidate features, linear scoring policy with hard masking, Adam, rank shaping |
| `meridian/metrics.py` | Battery-cycle metrics, episode runner with timing, paired statistics, log writers |
| `train.py` | Stage 1 behaviour cloning of JIT, stage 2 evolution strategies on daily profit |
| `evaluate.py` | Test seeds 2000–2199, fixed instance, sensitivity sweeps, auditor on every episode |
| `experiments/` | JIT rejection causes, failure analysis, rollout-teacher check |
| `run_all.sh` | Runs every step in order |
| `tests/test_suite.py` | Physics, fidelity to the baseline, determinism, masking, auditor fault injection |
| `results/` | Trained weights (`.npz`), training logs and curves, all evaluation outputs |

## Seeds

| Use | Seeds |
|---|---|
| Training days (demand × U[0.8, 1.3]) | 0 – 999 |
| Validation (model selection during training) | 5000 – 5029 |
| Test (all reported numbers) | 2000 – 2199 |
| Training RNG (`--seed`) | 0 |

Same seed + same actions → same trajectory (`test_determinism_same_seed_same_actions`).

## Ambiguities and how they were resolved

| # | Question | Choice |
|---|---|---|
| 1 | Where `Scenario.pdf` and `greedy_jit.py` could differ | `greedy_jit.py` rules 1–13 are followed exactly. The env's JIT mode reproduces the baseline schedule and KPIs exactly (tested on 5 configs × 15 seeds and on the fixed instance). |
| 2 | `ceil(d/3)` with float error (30.0/3) | Integer tenths of a km, as in the baseline. |
| 3 | A landing stint directly followed by a departure stint at the same vertiport | One merged stint in the output (Scenario §5). Charging is identical. |
| 4 | When is a decision made | At reveal time `max(0, earliest_dep − 30)`, in order of `(earliest_dep, id)`. No plan activity may start before reveal or before the aircraft's previous commitment ends. |
| 5 | Can a committed plan change later | No. Plans are immutable once accepted (no re-planning). |
| 6 | Where an aircraft waits | Parked (off-pad, no charging) between commitments. Extra charging is only possible inside a committed stint. |
| 7 | Day end | Every post-landing stint must end by t = 360. |
| 8 | Fleet sizes other than 8 | Initial positions cycle V0, V0, V1, V1, V2, V2, V3, V4 (baseline rule 11). |
| 9 | `charge_mult` | Scales both charging rates. |
| 10 | Battery "charge-discharge cycles" | Not defined in the spec. Reported as equivalent full cycles (EFC = SoC used / 100), charge EFC, number of charge→discharge reversals, and landings below 30 %. |
| 11 | Pad outage experiments | Modelled as a lower pad count for the whole day. A count of 0 means the vertiport is closed. |
| 12 | Extra charging modes (not in the spec) | PRECHARGE: start the pre-departure stint earlier while a pad is free, up to max(required SoC, 80 %). TOPUP: stay on the pad after landing while a pad is free, up to 80 %. LATE: latest feasible departure. All are checked exactly like JIT plans. |
| 13 | Auditor causality rule | Activities between an aircraft's previous revenue landing stint and a revenue departure must start at or after that request's reveal time. |

## Hardware used

See `results/policy_train_meta.json` (written by `train.py`). The numbers in
REPORT.md were produced on a 2-vCPU Intel Xeon (2.1 GHz) Linux VM with 8 GB RAM.
