# Previous training runs

This folder holds the results of the first two full training runs (seed 0), kept for comparison and for the presentation. The training pipeline never reads or changes this folder; new runs go to `models/stageN-seed0/` and `models/showcase/`.

Each run folder contains, per stage:

- `best.pt` (the best validated model);
- `training.jsonl` (every cloning round and PPO update);
- `evaluations.jsonl` (every validation);
- `best.json`, `config.json`, and `finished.json` when the stage completed.

`showcase/` holds the chosen model, its held-out test against the baseline (`SUMMARY.md`, `summary.json`, `test_*.json`) and the training history behind it. `latest.pt` and the intermediate `checkpoints/` are left out: they are only needed to continue an unfinished run, and they would add about 100 MB.

## Results

| | Run 1 (pipeline v1) | Run 2 (pipeline v2) | Hand-coded baseline |
| --- | --- | --- | --- |
| Model format / network width | swarmflow-v3 / 64 | swarmflow-v4 / 128 | – |
| Stage 1 best validation score | 0.997 | 0.996 | – |
| Stage 2 best validation score | 0.788 (then collapsed to about 0.2) | 0.956 (stable) | – |
| Stage 3 best validation score | 0.496 | 0.861 (then swung between 0.57 and 0.86) | – |
| Held-out test: episodes without failure | 138/160 | 148/160 | 160/160 |
| Held-out test: disturbances recovered | 38/80 | 66/80 | 80/80 |
| Held-out test score | 0.513 | 0.815 | 0.962 |
| Slot RMSE when undisturbed (hold / follow / turn) | 0.15 / 0.19 / 0.19 m (pass) | 0.08 / 0.05 / 0.08 m (pass) | 0.014 / 0.014 / 0.033 m |
| Verdict | NOT LEARNED YET | NOT LEARNED YET | – |

Where the runs failed:

- **Run 1.** The network was never shown obstacle avoidance, because cloning ran only in stage 1. Exploration had collapsed, all drones shared each crash penalty, and the obstacle warning started too late. Double crossing succeeded 0% of the time. Diagnosis and fixes: `training.md` Part C.
- **Run 2.** All 12 test failures were 1–2 cm grazes: slalom 8, head-on 2, double crossing 2. The teacher itself passes objects with only 8 cm to spare, so small imitation errors turned into contact. Diagnosis, the experiments and the fixes (version 3: a more cautious teacher, a stronger imitation loss, slalom removed): `training.md` Part D.

`run1-pipeline-v1/terminal_log_tail.txt` is the end of run 1's terminal output.

## Viewing a previous model

```sh
# macOS
.venv/bin/python -m swarmflow view --checkpoint models/previous-runs/run2-pipeline-v2/showcase/best.pt --port 8766
# Windows
.venv\Scripts\python -m swarmflow view --checkpoint models/previous-runs/run2-pipeline-v2/showcase/best.pt --port 8766
```

Open <http://127.0.0.1:8766>. The "Did the model learn correctly?" panel shows that run's curves, checks and learned-vs-baseline table. Replace `run2-pipeline-v2` with `run1-pipeline-v1` to see the first run.
