# BTP1 run

- mode: **train**
- policy: rl
- checkpoint: `D:\Presonal Project\BTP1_v1\models\latest.pt`
- seed: 0
- dumped: 20260921_023101

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 1 | False | 61.252095991716914 | collision_or_bounds |
| 2 | False | 40.08103960094485 | hold_failed |
| 3 | False | 84.70159561770919 | collision_or_bounds |

Zip this folder (`results/submit`) for submission.
