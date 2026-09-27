# BTP1 run

- mode: **train**
- policy: rl
- checkpoint: `D:\Presonal Project\BTP1_v1\models\latest.pt`
- seed: 0
- dumped: 20260921_022818

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 1 | False | 386.95349850908644 | collision_or_bounds |
| 2 | False | 457.11681562405965 | collision_or_bounds |
| 3 | False | 498.851675153801 | collision_or_bounds |

Zip this folder (`results/submit`) for submission.
