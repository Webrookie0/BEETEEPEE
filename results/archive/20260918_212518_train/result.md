# BTP1 run

- mode: **train**
- policy: rl
- checkpoint: `D:\Presonal Project\BTP1_v1\models\latest.pt`
- seed: 0
- dumped: 20260918_212518

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 1 | False | 0.5645048076507514 | collision_or_bounds |
| 2 | False | 0.7326868463483218 | hold_failed |
| 3 | False | 2.0682040755361597 | collision_or_bounds |

Zip this folder (`results/submit`) for submission.
