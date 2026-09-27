# BTP1 run

- mode: **baseline**
- policy: baseline
- checkpoint: `None`
- seed: 0
- dumped: 20260919_070109

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 3 | False | 30.307139904982872 | hold_failed |

Zip this folder (`results/submit`) for submission.
