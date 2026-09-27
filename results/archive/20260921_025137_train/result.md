# BTP1 run

- mode: **train**
- policy: rl
- checkpoint: `D:\Presonal Project\BTP1_v1\models\latest.pt`
- seed: 0
- dumped: 20260921_025137

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 1 | False | 368.82596399137685 | collision_or_bounds |
| 2 | False | 356.69195964975165 | collision_or_bounds |
| 3 | False | 362.1558387106529 | collision_or_bounds |

Zip this folder (`results/submit`) for submission.
