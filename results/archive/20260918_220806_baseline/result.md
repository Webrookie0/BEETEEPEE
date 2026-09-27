# BTP1 run

- mode: **baseline**
- policy: baseline
- checkpoint: `None`
- seed: 0
- dumped: 20260918_220806

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 1 | True | 0.30327336020749385 | ok |
| 2 | True | 0.04496240659654404 | ok |
| 3 | False | 0.9346283423218787 | collision_or_bounds |

Zip this folder (`results/submit`) for submission.
