# BTP1 run

- mode: **prod**
- policy: rl
- checkpoint: `D:\Presonal Project\BTP1_v1\models\best.pt`
- seed: 0
- dumped: 20260919_004820

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 2 | False | 0.40654695523258183 | short_episode |

Zip this folder (`results/submit`) for submission.
