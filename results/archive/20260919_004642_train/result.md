# BTP1 run

- mode: **train**
- policy: rl
- checkpoint: `D:\Presonal Project\BTP1_v1\models\latest.pt`
- seed: 0
- dumped: 20260919_004642

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 2 | False | 3.7968383854149566 | collision_or_bounds |

Zip this folder (`results/submit`) for submission.
