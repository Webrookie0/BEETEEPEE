# BTP1 run

- mode: **train**
- policy: rl
- checkpoint: `D:\Presonal Project\BTP1_v1\models\latest.pt`
- seed: 0
- dumped: 20260921_023327

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 1 | False | 62.02518748272907 | hold_failed |
| 2 | False | 54.279352067547414 | hold_failed |
| 3 | False | 284.0857144365852 | collision_or_bounds |

Zip this folder (`results/submit`) for submission.
