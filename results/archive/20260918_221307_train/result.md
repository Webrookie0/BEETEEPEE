# BTP1 run

- mode: **train**
- policy: rl
- checkpoint: `D:\Presonal Project\BTP1_v1\models\latest.pt`
- seed: 0
- dumped: 20260918_221307

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 1 | False | 0.3347242988450478 | hold_failed |
| 2 | False | 0.11247848140990722 | hold_failed |
| 3 | False | 0.8647658703552403 | hold_failed |

Zip this folder (`results/submit`) for submission.
