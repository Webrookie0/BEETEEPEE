# BTP1 run

- mode: **train**
- policy: rl
- checkpoint: `D:\Presonal Project\BTP1_v1\models\latest.pt`
- seed: 0
- dumped: 20260918_220548

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 1 | False | 2.6123722728277485 | collision_or_bounds |
| 2 | False | 3.039616954972688 | collision_or_bounds |
| 3 | False | 2.388480802842305 | collision_or_bounds |

Zip this folder (`results/submit`) for submission.
