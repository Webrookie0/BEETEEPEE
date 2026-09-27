# BTP1 run

- mode: **baseline**
- policy: baseline
- checkpoint: `None`
- seed: 0
- dumped: 20260918_220650

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 1 | False | 0.4129866052653279 | hold_failed |
| 2 | True | 0.04496240659654404 | ok |
| 3 | False | 1.0051422416487932 | collision_or_bounds |

Zip this folder (`results/submit`) for submission.
