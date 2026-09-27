# BTP1 run

- mode: **baseline**
- policy: baseline
- checkpoint: `None`
- seed: 0
- dumped: 20260919_065336

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 1 | True | 17.493054853179665 | ok |
| 2 | True | 1.304180971676753 | ok |
| 3 | True | 25.62059722478112 | ok |

Zip this folder (`results/submit`) for submission.
