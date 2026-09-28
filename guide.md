# Training guide: how to train, store and show the model

This guide lists the exact commands to train the drone policy yourself, store the results, and show the best model in the viewer. It also lists what **not** to do. [training.md](training.md) explains how the training works and why.

Run every command from the project folder (`BTP/Drone-Swarm`). On Windows, where `make` is usually missing, use the Python command given next to each `make` command, with `.venv\Scripts\python` instead of `.venv/bin/python`.

---

## 1. One-time setup

```sh
make install          # uv sync --python 3.12 --extra dev --cache-dir .uv-cache
make test             # about 10 s; every test must pass before you train
```

## 2. Train: one command

There are two ways to start the full training. Both run the same training and store the same results.

| Command | What it does | Python equivalent |
| --- | --- | --- |
| `make training` | Full training in the terminal: stage 1 → 2 → 3, then the showcase test. | `.venv/bin/python -m swarmflow pipeline` |
| `make run train` (same as `make run-train`) | The same training, **plus the viewer** in your browser. It shows live progress, flies the best model found so far, and switches to the final best model at the end. | `.venv/bin/python -m swarmflow pipeline --view` |

On a Mac, put `caffeinate -i` in front so the laptop does not sleep during the roughly 6-hour run:

```sh
caffeinate -i make run train
```

### What happens, in order

1. **Stage 1, formation keeping** (hold, follow, turn), saved to `models/stage1-seed0/`:
   - behavior cloning from the hand-coded controller, with 3 DAgger rounds;
   - 1,000,000 PPO steps.
2. **Stage 2, moving obstacles** (adds crossing and head-on objects), saved to `models/stage2-seed0/`:
   - starts from stage 1's `best.pt`;
   - behavior cloning + DAgger on these obstacle scenarios, so the network learns avoidance from the teacher;
   - 2,000,000 PPO steps with an imitation loss that fades out over the stage.
3. **Stage 3, the full benchmark** (adds repeated crossings and wind), saved to `models/stage3-seed0/`: the same recipe, starting from stage 2's `best.pt`, 2,000,000 PPO steps. Slalom, with static spheres on the formation slots, is not part of the learned-policy benchmark; see [training.md](training.md) Part D.
4. **Showcase**:
   - picks the best final model (across all trained seeds);
   - tests it against the hand-coded baseline on 7 scenarios × 20 seeds it has never seen;
   - stores everything in `models/showcase/`.

Every 100,000 steps the model is validated. Whenever it beats its previous best it is saved as `best.pt`, and each stage continues from the previous stage's `best.pt`.

### What you see while it runs

- **Terminal:** one line per update:
  - progress %, speed and ETA;
  - reward, share of failed episodes and slot error (last 100 episodes);
  - KL and value loss.

  Every validation prints its score and says when a new `best.pt` was saved.
- **Browser (`make run train`):** the section **"Did the model learn correctly?"** refreshes every 10 s:
  - a progress bar (stage, %, steps/s, time left in the stage, last and best validation score);
  - training curves for every stage;
  - the drones flown by the best model saved so far.

  Press **Run** to watch them. After the pipeline finishes, the page shows the final verdict and the learned-vs-baseline table, and the viewer keeps running until you press Ctrl+C.

### Time on this Mac (measured throughput)

| Part | Time |
| --- | --- |
| Stage 1 (cloning ~8 min + 1M PPO steps) | about 1 h |
| Stage 2 (cloning on obstacle scenarios ~15 min + 2M PPO steps) | about 2 h |
| Stage 3 (the same on all 7 scenarios) | about 2 h |
| Showcase test | about 10 min |
| **Total** | **about 6 h** |

This is version 3 of the training. [training.md](training.md) Part C covers the fixes after the first full run, and Part D covers the second run: a more cautious teacher, a stronger imitation loss, and slalom removed. When you start it, the older runs' folders are moved to `models/<name>-old-<date>` automatically, and everything trains again. Nothing is deleted.

### If it stops

Ctrl+C, a crash or the laptop sleeping loses nothing. Run the **same command again**:

- finished stages are skipped;
- the unfinished stage continues from its `latest.pt`, with its original settings and remaining steps.

### More seeds (recommended for the report)

```sh
caffeinate -i make train-full-multi        # seeds 0 1 2, about 18 h  (python -m swarmflow pipeline --seeds 0 1 2)
make training SEED=1                       # or add one seed at a time
```

The showcase always keeps the best seed. Report the others too, because RL results vary between seeds.

## 3. Where the results are stored

```text
models/
  stage1-seed0/ stage2-seed0/ stage3-seed0/   one folder per stage and seed
    best.pt               best validated checkpoint of that stage
    latest.pt             last update, only for continuing an interrupted run
    checkpoints/          snapshots every 500,000 steps
    training.jsonl        every cloning round and PPO update
    evaluations.jsonl     every validation (the learning curve)
    best.json             metrics of best.pt
    config.json           all settings;  finished.json appears when the stage is done
  showcase/                                   THE RESULT used by the viewer and the presentation
    best.pt               the selected model
    SUMMARY.md            verdict, per-stage table, acceptance checks, learned vs baseline
    summary.json          the same, machine-readable
    test_learned.json, test_baseline.json, test_compare.json   held-out test reports
    history/              training logs of every stage behind the selected model
  showcase-archive/<date>/                    the previous showcase, each time a new one is built
```

`models/best.pt` and `models/latest.pt` directly inside `models/` were not produced by this code, and the viewer ignores them.

## 4. Show the result

```sh
make run              # or double-click "Launch SwarmFlow.command"   (python -m swarmflow view)
```

Open <http://127.0.0.1:8765>. The viewer uses the best model automatically: `models/showcase/best.pt`, or, before the showcase exists, the best stage checkpoint. It keeps checking `models/`, so a newer best model is picked up without restarting.

1. **Active controller** reads `learned checkpoint`, and the yellow note shows which file is flying.
2. Pick a scenario and press **Run**. Only the network flies the drones; no baseline or safety filter runs behind it.
3. Switch **Controller** to *Predictive baseline* and run the same seed to compare. *Slot-only* shows what happens with no avoidance.
4. **"Did the model learn correctly?"**:
   - **Verdict**: `LEARNED CORRECTLY` only if every acceptance check passed on the held-out seeds (no failures; at least 95% of disturbances recovered within 5 s; slot error below 0.35 m on hold, follow and turn).
   - **Curves**: a rising validation score and falling failures show learning; ◆ marks the saved `best.pt`.
   - **Table**: learned policy vs baseline on the held-out seeds.

Other viewer commands: `make view-baseline` (no trained model), and `python -m swarmflow view --checkpoint <file> --port 8766` for one fixed checkpoint.

## 5. Stage by stage (optional)

These use the same settings as `make training`:

```sh
make train-stage1     make eval-stage1        # python -m swarmflow train --stage 1
make train-stage2     make eval-stage2        # starts from models/stage1-seed0/best.pt
make train-stage3     make eval-stage3        # starts from models/stage2-seed0/best.pt
make showcase                                 # choose, test on held-out seeds, store in models/showcase
make continue-stage2                          # finish an interrupted stage
```

The numbered tests `make test-9`, `test-10` and `test-11` compare each stage's `best.pt` with the baseline on the same 20 seeds. They reuse trained models and only train when one is missing.

## 6. Do and don't

**Do**

- Run `make test` once before a long training run.
- Use `caffeinate -i` on a Mac and keep the charger connected.
- Re-run the same command after an interruption; it continues by itself.
- Judge the model by the **viewer verdict**, the **validation curve** and `models/showcase/SUMMARY.md`.
- Train more than one seed for the report (`make train-full-multi`).
- Zip `models/showcase/` to share the result; it holds the model, its test reports and its training history.

**Don't**

- **Don't delete or edit files in a `models/stageN-seedS/` folder while it trains.** `latest.pt` is needed to continue, and `config.json` and `best.json` drive the resume and the model choice.
- **Don't run two trainings into the same folder at the same time**, for example `make training` in two terminals. Train different seeds instead (`SEED=1`).
- **Don't change `--drones`, `--duration`, `--noise` or `--dropout` between stages.** A stage can only continue from a checkpoint with the same environment settings.
- **Don't start a stage before the previous one has finished.** Stage 2 starts from stage 1's `best.pt`; `make training` handles the order for you.
- **Don't judge the model by training reward or loss alone.** Training uses exploration noise; the validation score and the held-out verdict are the real measures.
- **Don't tune anything on the held-out test seeds (100000+).** They are only for the final showcase test; use seeds 1000+ (`make eval-stageN`) while experimenting.
- **Don't try to resume old checkpoints from `runs/`** (format `swarmflow-v2`). They were trained with the old broken reward and network. They can still be *viewed* with `--checkpoint`, but not continued.
- **Don't shorten the stage budgets to make training finish faster.** The earlier 2–5 minute runs are exactly what failed (see section 2 of training.md).
- **Don't show `models/best.pt` directly inside `models/` as our result.** It was not produced by this code.

## 7. Troubleshooting

| Message | Meaning and fix |
| --- | --- |
| `... already holds a trained model; use --continue` | That folder already has a run. Use `make training` (it continues automatically), `make continue-stageN`, or another `--output`. |
| `Stage 2 starts from models/stage1-seed0/best.pt; train stage 1 first` | Run the stages in order, or just use `make training`. |
| `... is a swarmflow-v2 checkpoint trained with the old reward and network` | An old run cannot be resumed; train fresh. |
| `Resume checkpoint has incompatible environment configuration` | The environment settings differ from the checkpoint; use the same ones in every stage. |
| `Port 8765 is already in use` (with `make run train`) | A viewer is already running; it follows the training too. Or close it and run again. |
| Verdict `NOT LEARNED YET` | Read the failing check and the curves, then see section 6.3 of [training.md](training.md). |
