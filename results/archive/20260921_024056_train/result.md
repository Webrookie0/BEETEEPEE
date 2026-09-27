# BTP1 run

- mode: **train**
- policy: rl
- checkpoint: `D:\Presonal Project\BTP1_v1\models\latest.pt`
- seed: 0
- dumped: 20260921_024056

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 1 | False | 559.3650736001714 | collision_or_bounds |
| 2 | False | 414.9810919009134 | collision_or_bounds |
| 3 | False | 598.3529280377181 | collision_or_bounds |

Zip this folder (`results/submit`) for submission.
