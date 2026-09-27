# BTP1 run

- mode: **train**
- policy: rl
- checkpoint: `D:\Presonal Project\BTP1_v1\models\latest.pt`
- seed: 0
- dumped: 20260918_211605

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 1 | False | 5.765952159407723 | collision_or_bounds |
| 2 | False | 6.503295713220659 | collision_or_bounds |
| 3 | False | 6.01846750840387 | collision_or_bounds |

Zip this folder (`results/submit`) for submission.
