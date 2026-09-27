# BTP1 run

- mode: **train**
- policy: rl
- checkpoint: `D:\Presonal Project\BTP1_v1\models\latest.pt`
- seed: 0
- dumped: 20260921_025852

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 1 | False | 45.20593829734949 | hold_failed |
| 2 | False | 172.24062638830387 | hold_failed |
| 3 | False | 144.64177397707786 | hold_failed |

Zip this folder (`results/submit`) for submission.
