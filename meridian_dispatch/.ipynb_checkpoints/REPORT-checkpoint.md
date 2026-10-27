# Meridian eVTOL dispatch: short note

> **Status of the numbers in this note.** Sections 1–4 and 6–7 describe the
> design and are final. The numbers marked **[verified]** were produced in the
> development run. The full test evaluation (200 held-out days, sensitivity
> sweeps, failure analysis) and the reward-variant training were **not run
> before packaging**. Run `bash run_all.sh` (or the steps in README.md). The
> scripts write every table to `results/tables.md`, `results/failure_analysis.md`
> and `results/jit_rejection_causes.md`. Copy those tables into Section 5.

## 1. Summary

* **Environment.** An event-driven simulator with one decision per request.
  Its JIT mode reproduces `greedy_jit.py` exactly: same schedule and same KPIs
  on the fixed instance, and on 15 seeds for each of 5 configurations
  (default, fleet 6, fleet 10, charge ×0.75, pads 2,2,2,1,1) **[verified, tests]**.
* **Independent checker.** `meridian/auditor.py` imports only the Python
  standard library. It rebuilds the whole day from the CSV logs and checks 14
  rules. Fault-injection tests show that it catches pad, reserve, unserved
  request, rejected-but-flown, time window, block time, short stint, continuity
  and reveal-time violations **[verified, tests]**.
* **Policy.** A linear scorer over hand-built features of each candidate plan,
  with hard action masking. Training: behaviour cloning of JIT, then evolution
  strategies (ES) on the true daily profit. Training took **17.2 min on a 2-core
  Intel Xeon 2.1 GHz VM** **[verified]**, well under the 45-minute limit.
* **Validation result** (30 days, seeds 5000–5029, nominal demand) **[verified]**:

  | Policy | Mean daily profit |
  |---|---|
  | Greedy-JIT | $4,165.0 |
  | JIT + precharge heuristic (hand-written) | $4,346.0 (+4.3 %) |
  | Learned policy (best checkpoint) | $4,553.9 (+9.3 %) |

  These are validation numbers and were used for model selection. The
  unbiased comparison is the test set (seeds 2000–2199) in `results/tables.md`.
* **Safety.** No accepted request can fail and no flight can break the 20 %
  reserve, because the policy can only choose among plans that were fully
  checked before they were offered. The auditor confirms this on every
  evaluated episode.

## 2. Problem A: simulation environment

### 2.1 Event-driven decision process

The day is a sequence of decision epochs, one per request, in order of
`(earliest_dep, id)`. The epoch time is the reveal time
`max(0, earliest_dep − 30)`. This matches the scenario: the operator must
accept or reject when a request becomes known, and committed plans are never
changed. Between epochs, nothing needs to be simulated, because all aircraft
movements are already fixed by earlier commitments. This makes the process a
semi-Markov decision process with variable time between decisions.

**State** at an epoch: the new request; for each aircraft its location, the
time it becomes free and its SoC; the minute-by-minute pad occupancy of every
vertiport (0–360).

**Actions:** `0 = REJECT`, or `1 + m·K + k` = "serve with aircraft k using
charging mode m" (K = fleet size, 4 modes).

### 2.2 Candidate plans and charging modes

For each aircraft and mode, the planner builds one complete plan (stints,
optional deadhead, revenue flight, post-landing stint) and checks it fully.

| Mode | What changes compared with JIT |
|---|---|
| JIT | Nothing. Exact copy of `plan_for_aircraft` in `greedy_jit.py`. |
| PRECHARGE | The pre-departure (or pre-deadhead) stint starts earlier, minute by minute, while a pad is free and the aircraft is free, up to max(required SoC, 80 %). Departure time is unchanged. |
| TOPUP | The post-landing stint continues while a pad is free, up to 80 % SoC. |
| LATE | Latest feasible departure in the window instead of the earliest. |

Why these modes: a diagnostic on JIT (`experiments/jit_rejection_causes.py`)
showed that most forced rejections are caused by **low battery**, not by pads.
On the 200 test days JIT had 6,129 forced rejections. Giving every aircraft
100 % SoC would make 79 % of them servable, unlimited pads 52 %, and a free
aircraft 70 % **[verified, `results/jit_rejection_causes.md`]**. JIT charges
only the minimum and then parks the aircraft without charging, so an aircraft
is often near the reserve when the next request arrives. The modes give the
policy a way to charge when pads are idle. A plan that is identical to another
mode's plan is masked as a duplicate.

### 2.3 Requirements checklist

| Requirement | How it is met |
|---|---|
| Energy and 20 % reserve | Every flight's SoC is computed exactly (floats, never rounded). A plan is built only if the SoC before each flight is at least 20 + 12 + d. |
| Charging curve | 2 %/min below 80 %, 2/3 %/min above, cap 100 %, scaled by `charge_mult`. |
| Pad capacity | Minute-level pad book; a plan is offered only if every stint fits. |
| Time windows | Departure in [earliest_dep, latest_dep]; post-landing stint must end by 360. |
| 30-minute lead time | No activity of a new plan starts before the reveal time. |
| Fixed instance and generator | `reset(requests=...)` or `reset(seed=..., demand_mult=..., peak_shift=...)`. The generator uses the same random stream as the baseline and gives identical requests for the same seed (tested). |
| Deterministic seeding | Same seed + same actions → identical schedule and KPIs (tested). Training is also deterministic: two runs with seed 0 gave the same validation curve. |
| Independent checker | `meridian/auditor.py`, see 2.4. |

### 2.4 Independent constraint checker

The auditor reads only `requests.csv`, `schedule.csv`, `decisions.csv` and a
small config JSON. It retypes the vertiport coordinates from the scenario,
recomputes distances, block times, energy and charging with its own code, and
replays each aircraft from its start position at 90 % SoC. It shares no code
with the environment or the baseline (a test checks its imports).

| Code | Rule |
|---|---|
| D1 | One decision per request, correct reveal time |
| F1 | Every ACCEPT has exactly one revenue flight (otherwise: service failure) |
| F2 | No REJECTed or unknown request is flown |
| F3 | Revenue flight has the booked origin/destination and departs inside the window |
| T1 | Flight duration = 4 + ceil(d/3) |
| T2 | Activities inside [0, 360]; stints ≥ 5 min |
| C1, C2, C3 | Start position; no overlap; no teleporting; every departure ends a stint and every landing starts a stint |
| E1, E2 | Landing SoC ≥ 20 %; SoC ≤ 100 % |
| P1 | Pad occupancy ≤ pad count at every minute |
| R1 | Activities that prepare a revenue flight start at or after its reveal time |
| K1 | KPIs recomputed from the schedule match the reported KPIs |

On the fixed instance, the auditor passes the JIT output and recomputes its
profit exactly ($4,011.64) **[verified]**.

## 3. Problem B: policy design

### 3.1 Policy structure

Each unmasked action gets a feature vector φ(s, c) with 111 entries:

* plan features (deadhead km, plan profit, departure delay, SoC at departure
  and at the end, minimum landing SoC, how long the aircraft is busy, stint
  lengths, whether it leaves its vertiport without an aircraft, how many
  aircraft are already at the destination, pad use at the destination after
  landing and at the origin);
* mode one-hot, and products of the plan features with the mode (so the same
  feature can have a different effect per mode, for example a long
  post-landing stint is cheap at V0 with 3 pads and harmful at V3 with 1 pad);
* context features (time, expected demand, request margin, number of feasible
  aircraft, idle fleet share, fleet SoC, route one-hots, pad scarcity). These
  are placed on the REJECT row, so the reject threshold depends on context.

The score is `w · φ`. The policy takes the arg-max over **unmasked** actions only.

Two choices are worth explaining:

1. **One shared scorer instead of a fixed 25-way output.** The weights do not
   depend on the number of aircraft, so the same trained policy runs with 6, 8
   or 10 aircraft. A fixed output head per aircraft index cannot be evaluated on
   a larger fleet without retraining.
2. **Linear instead of an MLP.** I first trained a 2-layer MLP (40→32→16→1).
   After behaviour cloning, its hidden units were saturated: small ES noise
   changed no decision, and large noise flipped almost all decisions at once
   (for example, every request rejected). ES then had no useful signal. A
   linear scorer with mode interactions gives graded changes, and ES learned
   steadily. I also removed a feature "is this the JIT choice", because it made
   every score gap so large that the same problem appeared.

### 3.2 Safety: action masking, not a shield or a penalty

The safety rule is enforced **by construction**. The environment offers only
candidates that passed every check (reserve, pads, windows, reveal time, day
end). Infeasible candidates are masked, and `env.step` raises an error if a
masked action is passed. REJECT is always available, so there is always a legal
action. Because every plan is committed in full at decision time and never
changed, an accepted request cannot fail later. A test runs a random policy over
all unmasked actions (all modes, three configurations, 20 days each) and the
auditor finds no violation.

Why not the alternatives:

* **Penalty in the reward.** A penalty only makes violations unlikely during
  training. It does not rule them out. Under distribution shift (more demand,
  slower chargers) the policy meets states it never saw, and a learned
  trade-off between profit and penalty can choose a violation. The size of the
  penalty is also a tuning problem: too small and violations pay; too large and
  the policy rejects too much. With ES, a large penalty also makes the return
  landscape very uneven.
* **Shield that corrects an unsafe action afterwards.** A shield maps an
  unsafe choice to a safe one (for example, reject). The policy is then trained
  on actions that were not the ones executed, which biases learning. The
  correction rule is also a hidden second policy that can behave badly (it may
  reject many good requests). Masking gives the same guarantee, but the policy
  only ever chooses among safe options and learns their true effect.
* **Masking needs an exact model.** It is only as good as the feasibility check.
  This is why the independent auditor matters: it re-checks every episode from
  the raw logs, with separate code.

### 3.3 Reward design

**Main reward:** at each decision, the exact profit change of the action:
`r = revenue − 1.5·(d_rev + d_dh) − 0.3·(E_rev + E_dh)` if accepted, 0 if
rejected. The rewards of a day sum exactly to the day's profit (tested). There
is no discount, because the day is short and finite.

ES optimises the total return of a day, so dense versus terminal-only reward
makes no difference for ES. This is one reason ES was chosen: no credit
assignment design is needed.

**Variants considered and rejected:**

1. **Acceptance bonus (+B per served request), tested.** This rewards service
   level. It values a request by count, not by margin, so it pushes the policy
   to accept plans with long deadheads and to take requests that use up an
   aircraft needed for a better request later. `train.py --accept-bonus 50`
   trains this variant. Model selection uses the variant's own reward (not the
   true profit), so the comparison is fair. In a partial run (45 of 200
   iterations, stopped when packaging), validation return rose fast because of
   the bonus, while its true profit on training days stayed around the main
   policy's level. The full comparison is produced by `run_all.sh` step 3 and
   `evaluate.py` (policy "RL-bonus").
2. **Shaping terms (bonus for SoC gained in the fast region, penalty for pad
   congestion at the destination), considered.** Both are not potential-based,
   so they change the optimal policy. The SoC bonus rewards charging even when
   the charge is never used, and occupies pads. The policy already sees SoC and
   pad use as features, so the true profit is enough to learn when they matter.
3. **Rollout teacher, tried.** A Monte-Carlo teacher scored each action by
   immediate profit plus the profit of a base policy on 4 sampled futures. On 3
   test days it was **worse** than JIT (mean $3,308 vs $3,520), at about 12 s
   per day **[verified, 3 days]**. With few samples, the maximum over about 9 noisy
   estimates is biased upwards (the optimiser's curse); the teacher rejected
   8–10 feasible requests per day. More samples would fix this but cost too
   much time. `experiments/rollout_teacher_check.py` reruns it on 10 days.

### 3.4 Training

| Stage | What | Data |
|---|---|---|
| 1. Behaviour cloning | Softmax cross-entropy towards the JIT choice, Adam, L2 1e-4; stops at 99.9 % accuracy or 25 epochs | 150 JIT days, about 8,400 decisions |
| Calibration | Rescale weights so the median score gap is 1, so ES noise of σ = 0.3 changes some decisions but not all | — |
| 2. Evolution strategies | 200 iterations, 16 antithetic pairs, σ = 0.3, Adam lr 0.05, centred ranks (ties share a rank), L2 1e-4 | 8 training days per iteration, demand × U[0.8, 1.3], seeds 0–999, the same days for all population members (common random numbers) |
| Model selection | Every 5 iterations on validation seeds 5000–5029 | — |

BC reached 86.7 % accuracy. The rest are mostly ties: plans that are identical
except for the aircraft index, which JIT breaks by index. The BC policy's
validation profit was $4,170.6 versus $4,165.0 for JIT, so stage 1 starts at
the baseline.

Why ES: the decision is discrete (arg-max over a variable set), the reward
arrives per day, and the episode is cheap (~35 ms). ES needs no gradient through
the arg-max, no value function and no credit assignment. It parallelises
across days, and common random numbers remove most of the day-to-day noise.
PPO or DQN would need a value network over a variable number of candidates and
careful tuning; they were not needed to beat the baseline.

Hardware: 2 vCPU Intel Xeon @ 2.1 GHz (KVM), 8 GB RAM, Python 3.13, numpy
only. Total time 17.2 min (BC 12 s). Details: `results/policy_train_meta.json`,
learning curve `results/policy_curve.csv`.

## 4. Evaluation framework

* **Data split.** Training seeds 0–999, validation 5000–5029 (model selection
  only), test 2000–2199 (all reported numbers). They do not overlap.
* **Paired comparison.** Every policy runs on the same 200 test days. The
  table reports mean ± SD, median, mean difference to JIT with 95 % CI, share of
  days better/worse, paired t-test and Wilcoxon signed-rank p-values.
* **Baselines.** Greedy-JIT, and a hand-written "JIT + precharge" heuristic.
  The second baseline checks if the learned policy does more than a simple rule.
* **Metrics.** Profit; requests served; rejections split into forced (no
  feasible plan) and strategic (feasible but declined); service failures
  (from the auditor); deadhead km per day and per served request; battery use;
  pad utilisation per vertiport; minimum landing SoC; inference time per day and
  per decision.
* **Battery cycles.** Not defined in the scenario. Reported as: discharge
  equivalent full cycles = Σ SoC used in flight / 100 (fleet, per day); charge
  EFC = Σ SoC added / 100; charge→discharge reversals; landings below 30 % SoC
  (deep discharge, the main ageing driver).
* **Auditor on everything.** Every episode of every policy and setting is
  re-checked by the auditor. The fixed-instance logs are written to
  `results/fixed/` and audited from the files.
* **Sensitivity (no retraining).** Demand × {0.7, 0.9, 1.1, 1.3, 1.6, 2.0};
  peak shift {−45, −30, −15, +15, +30, +60} min; charge rate ×
  {0.75, 0.9, 1.1, 1.5}; pads (2,2,2,1,1), (3,1,2,1,1), (3,2,1,1,1), (3,2,2,1,0);
  fleet 6 and 10. Small input noise: the fixed instance with every window
  shifted by U{−s..s} minutes, s ∈ {2, 5, 10}, 30 variants each.

## 5. Results

**To fill in from `results/tables.md` after running `evaluate.py`.**

Verified with the shipped weights (`results/failure_analysis.md`):

* **Test days 2000–2199, nominal:** the learned policy makes less profit than
  JIT on only **12 of 200 days**. The worst day (seed 2077) is −$148; the best
  day (seed 2049) is +$1,154.
* **Fixed 70-request instance:** the learned policy makes **less** profit than
  JIT: $3,973.24 (51 served, 90.1 deadhead km) against $4,011.64 (53 served,
  158.0 deadhead km). The precharge heuristic also loses here ($3,917.58). Both
  pass the auditor.
* **Behaviour:** it uses PRECHARGE for 34–48 % of accepted requests (most at V3,
  the busiest single-pad vertiport), never uses TOPUP or LATE, and declines
  7.05 feasible requests per day. Declined requests are cheap ones: median
  best-plan profit $45, and 49 % of them go to V3 (689 of 1,410).
* In a 5-day smoke run, inference took about 0.4 ms per decision (35 ms per
  day) against about 0.1 ms for JIT, because 4 modes × 8 aircraft plans are
  built at every step. Final timings are in `results/tables.md`.

The full table (mean ± SD, CI, p-values, served, deadhead, battery cycles, pad
use) and the sensitivity table are produced by `python evaluate.py`.

## 6. Failure cases and where this would be unsafe

`experiments/failure_analysis.py` lists the worst test days and the fixed
instance request by request. Failure modes found:

1. **Strategic rejection that does not pay.** The policy learns to decline
   some feasible requests to keep aircraft or charge for later. On one specific
   day the expected later request may not come, and the declined revenue is
   lost. On the fixed instance, RL declined R14 (V1→V3, best plan needed 30 km
   of deadhead, plan profit $25) and R59 (V0→V3, 24 km deadhead, $30), and 5 more
   requests became infeasible because of earlier choices. The aircraft it saved
   served 5 other requests worth $333, which was not enough. On test seed 2077,
   it declined 4 requests that JIT served and lost $148.
2. **Holding single pads.** TOPUP and PRECHARGE occupy a pad for longer. At V3
   and V4 (one pad each) this can block later landings. The mode interaction
   with destination pad scarcity is meant to learn this. The trained policy
   never chose TOPUP, so it seems to have learned that holding a pad after
   landing does not pay in this network.
3. **Distribution shift.** The policy saw demand × 0.8–1.3 in training. At
   ×2.0 demand, or with a closed vertiport, the learned reject threshold may be
   wrong. The features include the time of day and the *nominal* demand curve,
   not the real one, so a shifted peak is not observed directly.

**Unsafe to deploy as it is**, even though no simulated rule is ever broken:

* **The guarantee is only as good as the model.** Flight time, energy use and
  charging rate are exact and deterministic here. Real wind, payload,
  temperature and battery ageing change energy use and charge speed. A plan
  that lands at 20.1 % in the model can land below 20 % in reality.
  Deployment needs an energy margin above the reserve and a check at departure.
* **No re-planning.** Plans are fixed at acceptance. One late landing can keep
  a pad busy and break every later plan that counted on that pad (a cascade).
  A real system needs online re-planning and a rule for what to do when a
  committed plan becomes infeasible.
* **Charger and pad faults** are not modelled during a day. A fault after
  acceptance would cause service failures that this policy cannot avoid.
* **Demand model.** The policy was trained only on this Poisson generator. A
  different real demand pattern (events, weather) can make strategic rejections
  costly. It should be validated on real booking data and watched for drift.

## 7. AI-tool disclosure

I used Claude (Anthropic, Opus model) as a coding and writing assistant through
the claude.ai app with code execution. It wrote most of the first versions of
the code, the tests and this note from my instructions and the assignment
files, ran the experiments in a cloud VM, and reported the results. I directed
the design choices, reviewed the code and the results, and checked the
explanations. The diagnostic steps (why the MLP failed under ES, why the context
features had no effect at first, why the rollout teacher was worse) came from
running and inspecting the experiments in that session. 

System used to run code:
Device name	DESKTOP-0I2EQG2 \\
Processor	Intel(R) Core(TM) i9-14900 (2.00 GHz) \\
Installed RAM	16.0 GB (15.7 GB usable) \\
System type	64-bit operating system, x64-based processor \\
Pen and touch	Pen support
