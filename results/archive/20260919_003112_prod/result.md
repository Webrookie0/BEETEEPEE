# BTP1 run

- mode: **prod**
- policy: rl
- checkpoint: `D:\Presonal Project\BTP1_v1\models\best.pt`
- seed: 0
- dumped: 20260919_003112

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 2 | False | 0.9838614076181098 | collision_or_bounds |

Zip this folder (`results/submit`) for submission.
