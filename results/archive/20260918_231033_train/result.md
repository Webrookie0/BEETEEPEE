# BTP1 run

- mode: **train**
- policy: rl
- checkpoint: `D:\Presonal Project\BTP1_v1\models\latest.pt`
- seed: 0
- dumped: 20260918_231033

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 1 | False | 7.190841138617707 | collision_or_bounds |
| 2 | False | 7.223924047990171 | collision_or_bounds |
| 3 | False | 5.798905235745586 | collision_or_bounds |

Zip this folder (`results/submit`) for submission.
