# BTP1 run

- mode: **baseline**
- policy: baseline
- checkpoint: `None`
- seed: 0
- dumped: 20260919_014115

## Model files

- train dump: `models/latest.pt` (current net, overwritten every PPO update)
- prod dump: `models/best.pt` (best combined gate score)
- this pack copies the loaded checkpoint into `models/`

## Pass gates

| Test | passed | error | reason |
| --- | --- | --- | --- |
| 1 | True | 17.49302819967417 | ok |
| 2 | True | 2.2481373610449906 | ok |
| 3 | True | 22.4181666099075 | ok |

Zip this folder (`results/submit`) for submission.
