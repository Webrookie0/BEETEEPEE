# BTP1 run

- mode: **train**
- policy: rl
- checkpoint: `D:\Presonal Project\BTP1_v1\models\latest.pt`
- seed: 0
- dumped: 20260918_211413

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 1 | False | 7.069822811355396 | collision_or_bounds |
| 2 | False | 7.057859927763161 | collision_or_bounds |
| 3 | False | 8.670669818318158 | collision_or_bounds |

Zip this folder (`results/submit`) for submission.
