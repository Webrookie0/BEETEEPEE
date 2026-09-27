# BTP1 run

- mode: **baseline**
- policy: baseline
- checkpoint: `None`
- seed: 0
- dumped: 20260918_223735

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 2 | False | 1.6793829678728927 | collision_or_bounds |

Zip this folder (`results/submit`) for submission.
