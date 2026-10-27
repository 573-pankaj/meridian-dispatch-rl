# Causes of Greedy-JIT forced rejections (200 test days, 6129 rejections)

| Relaxed resource | Rejections that become servable |
|---|---|
| battery (all aircraft at 100 % SoC) | 4830 (79%) |
| pads (unlimited) | 3211 (52%) |
| aircraft time (no earlier commitment) | 4279 (70%) |

| Any single relaxation that works | Count |
|---|---|
| pads + soc + busy | 1783 |
| soc + busy | 1655 |
| pads + soc | 926 |
| busy | 607 |
| soc | 466 |
| pads | 268 |
| pads + busy | 234 |
| none of the three | 190 |
