# BTP1 run

- mode: **train**
- policy: rl
- checkpoint: `D:\Presonal Project\BTP1_v1\models\latest.pt`
- seed: 0
- dumped: 20260919_013921

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 1 | True | 5.857730298283345 | ok |
| 2 | False | 194.1945081011921 | collision_or_bounds |
| 3 | False | 100.49462477430875 | collision_or_bounds |

Zip this folder (`results/submit`) for submission.
