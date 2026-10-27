# Failure analysis: learned policy vs Greedy-JIT

## 1. How the learned policy behaves (200 test days, nominal)

- Accepted 11746, strategic rejections 1410 (7.05 per day). In 302 of the strategic rejections a plan with zero deadhead existed.
- Profit of the best plan the policy turned down: median 45.0, min 4.3, max 118.4 $.
- Strategic rejections by destination: V0: 153, V1: 142, V2: 143, V3: 689, V4: 283
- Strategic rejections by departure hour (0 = 06:00): 0: 92, 1: 316, 2: 697, 3: 193, 4: 112

| Destination (pads) | JIT mode | TOPUP | PRECHARGE | LATE |
|---|---|---|---|---|
| V0 (3) | 65% | 0% | 35% | 0% |
| V1 (2) | 65% | 0% | 35% | 0% |
| V2 (2) | 66% | 0% | 34% | 0% |
| V3 (1) | 52% | 0% | 48% | 0% |
| V4 (1) | 63% | 0% | 37% | 0% |

- RL loses to JIT on 12 of 200 days; worst day 2077 (-148.2 $), best day 2049 (+1153.6 $).

## 2. Worst test days for RL

### Seed 2077 (RL − JIT = -148.2 $)

JIT profit 4289.10 (58 served, 287.6 dh km); RL profit 4140.90 (55 served, 238.9 dh km, 5 strategic rejections). RL modes JIT/TOPUP/PRECHARGE/LATE = 30/0/25/0.

- Served by JIT only: 8 (RL rejected 4 of them by choice, 4 because no plan was feasible any more).
- Served by RL only: 5, with plan profit 273.

| Request | Route | Window | Best plan profit | Min deadhead km |
|---|---|---|---|---|
| R06 | V4→V2 | 29–44 | 5.0 | 37.2 |
| R20 | V2→V3 | 87–102 | 27.0 | 30.0 |
| R30 | V3→V1 | 123–138 | 46.1 | 18.4 |
| R44 | V0→V4 | 183–198 | 65.5 | 11.4 |

### Seed 2036 (RL − JIT = -143.9 $)

JIT profit 4256.76 (59 served, 287.7 dh km); RL profit 4112.86 (55 served, 195.5 dh km, 8 strategic rejections). RL modes JIT/TOPUP/PRECHARGE/LATE = 33/0/22/0.

- Served by JIT only: 10 (RL rejected 5 of them by choice, 5 because no plan was feasible any more).
- Served by RL only: 6, with plan profit 470.

| Request | Route | Window | Best plan profit | Min deadhead km |
|---|---|---|---|---|
| R07 | V4→V0 | 24–39 | 19.1 | 37.2 |
| R11 | V0→V3 | 31–46 | 39.5 | 19.0 |
| R22 | V0→V1 | 76–91 | 40.4 | 19.0 |
| R36 | V0→V1 | 139–154 | 30.9 | 24.3 |
| R53 | V0→V2 | 187–202 | 14.5 | 24.3 |

### Seed 2138 (RL − JIT = -135.6 $)

JIT profit 4228.98 (53 served, 115.2 dh km); RL profit 4093.34 (53 served, 153.3 dh km, 2 strategic rejections). RL modes JIT/TOPUP/PRECHARGE/LATE = 25/0/28/0.

- Served by JIT only: 5 (RL rejected 2 of them by choice, 3 because no plan was feasible any more).
- Served by RL only: 5, with plan profit 281.

| Request | Route | Window | Best plan profit | Min deadhead km |
|---|---|---|---|---|
| R24 | V0→V4 | 147–162 | 65.5 | 11.4 |
| R37 | V0→V4 | 176–191 | 89.7 | 0.0 |

## 3. Fixed 70-request instance

### data/requests.csv

JIT profit 4011.64 (53 served, 158.0 dh km); RL profit 3973.24 (51 served, 90.1 dh km, 3 strategic rejections). RL modes JIT/TOPUP/PRECHARGE/LATE = 36/0/15/0.

- Served by JIT only: 7 (RL rejected 2 of them by choice, 5 because no plan was feasible any more).
- Served by RL only: 5, with plan profit 333.

| Request | Route | Window | Best plan profit | Min deadhead km |
|---|---|---|---|---|
| R14 | V1→V3 | 66–81 | 25.2 | 30.0 |
| R59 | V0→V3 | 196–211 | 29.9 | 24.3 |

