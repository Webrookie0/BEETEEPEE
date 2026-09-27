# BTP1 run

- mode: **train**
- policy: rl
- checkpoint: `D:\Presonal Project\BTP1_v1\models\latest.pt`
- seed: 0
- dumped: 20260918_230216

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 1 | False | 0.33032784659507014 | collision_or_bounds |
| 2 | False | 0.5576213273725786 | hold_failed |
| 3 | False | 0.8528443478200318 | collision_or_bounds |

Zip this folder (`results/submit`) for submission.
