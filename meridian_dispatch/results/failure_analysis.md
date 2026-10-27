# Failure analysis: learned policy vs Greedy-JIT

## 1. How the learned policy behaves (200 test days, nominal)

- Accepted 11796, strategic rejections 1196 (5.98 per day). In 302 of the strategic rejections a plan with zero deadhead existed.
- Profit of the best plan the policy turned down: median 45.0, min 4.3, max 89.7 $.
- Strategic rejections by destination: V0: 49, V1: 106, V2: 83, V3: 735, V4: 223
- Strategic rejections by departure hour (0 = 06:00): 0: 86, 1: 267, 2: 572, 3: 173, 4: 98

| Destination (pads) | JIT mode | TOPUP | PRECHARGE | LATE |
|---|---|---|---|---|
| V0 (3) | 68% | 0% | 32% | 0% |
| V1 (2) | 67% | 0% | 33% | 0% |
| V2 (2) | 66% | 0% | 34% | 0% |
| V3 (1) | 51% | 0% | 49% | 0% |
| V4 (1) | 61% | 0% | 39% | 0% |

- RL loses to JIT on 14 of 200 days; worst day 2082 (-547.0 $), best day 2066 (+1364.6 $).

## 2. Worst test days for RL

### Seed 2082 (RL − JIT = -547.0 $)

JIT profit 4717.54 (61 served, 235.7 dh km); RL profit 4170.50 (56 served, 283.9 dh km, 1 strategic rejections). RL modes JIT/TOPUP/PRECHARGE/LATE = 37/0/19/0.

- Served by JIT only: 13 (RL rejected 1 of them by choice, 12 because no plan was feasible any more).
- Served by RL only: 8, with plan profit 485.

| Request | Route | Window | Best plan profit | Min deadhead km |
|---|---|---|---|---|
| R17 | V1→V3 | 80–95 | 45.0 | 19.0 |

### Seed 2077 (RL − JIT = -240.7 $)

JIT profit 4289.10 (58 served, 287.6 dh km); RL profit 4048.42 (55 served, 264.8 dh km, 7 strategic rejections). RL modes JIT/TOPUP/PRECHARGE/LATE = 30/0/25/0.

- Served by JIT only: 9 (RL rejected 5 of them by choice, 4 because no plan was feasible any more).
- Served by RL only: 6, with plan profit 323.

| Request | Route | Window | Best plan profit | Min deadhead km |
|---|---|---|---|---|
| R06 | V4→V2 | 29–44 | 5.0 | 37.2 |
| R08 | V0→V3 | 34–49 | 39.5 | 19.0 |
| R20 | V2→V3 | 87–102 | 27.0 | 30.0 |
| R30 | V3→V1 | 123–138 | 46.1 | 18.4 |
| R44 | V0→V4 | 183–198 | 65.5 | 11.4 |

### Seed 2000 (RL − JIT = -138.6 $)

JIT profit 4560.94 (60 served, 128.3 dh km); RL profit 4422.36 (59 served, 203.8 dh km, 8 strategic rejections). RL modes JIT/TOPUP/PRECHARGE/LATE = 37/0/22/0.

- Served by JIT only: 10 (RL rejected 8 of them by choice, 2 because no plan was feasible any more).
- Served by RL only: 9, with plan profit 686.

| Request | Route | Window | Best plan profit | Min deadhead km |
|---|---|---|---|---|
| R16 | V0→V4 | 74–89 | 42.3 | 24.3 |
| R17 | V0→V2 | 76–91 | 24.1 | 19.0 |
| R19 | V0→V3 | 84–99 | 39.5 | 19.0 |
| R21 | V0→V3 | 88–103 | 39.5 | 19.0 |
| R27 | V0→V2 | 103–118 | 14.5 | 24.3 |
| R45 | V4→V3 | 151–166 | 78.4 | 0.0 |
| R57 | V2→V3 | 178–193 | 41.2 | 22.1 |
| R67 | V2→V3 | 206–221 | 60.5 | 11.4 |

## 3. Fixed 70-request instance

### data/requests.csv

JIT profit 4011.64 (53 served, 158.0 dh km); RL profit 3907.98 (52 served, 167.6 dh km, 4 strategic rejections). RL modes JIT/TOPUP/PRECHARGE/LATE = 35/0/17/0.

- Served by JIT only: 3 (RL rejected 1 of them by choice, 2 because no plan was feasible any more).
- Served by RL only: 2, with plan profit 87.

| Request | Route | Window | Best plan profit | Min deadhead km |
|---|---|---|---|---|
| R14 | V1→V3 | 66–81 | 25.2 | 30.0 |

