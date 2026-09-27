# SwarmFlow

An independent reinforcement-learning environment for a drone swarm that follows a reference target, makes room for obstacles, and recovers the **same assigned formation slots** afterward.

Start with the [research and execution plan](docs/research-plan.md). It connects six primary research sources to concrete design decisions and distinguishes established results from hypotheses. The main recommendation is to learn swarm motion above a stable flight controller and study formation recovery explicitly.

## Setup and run

SwarmFlow runs locally on Python 3.12. From the project directory, install the environment and development dependencies:

```sh
cd /Users/sumit/Documents/Codex/developer/neww
make install
```

Start the interactive viewer with:

```sh
make run
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765). The default controller is the predictive classical baseline. To run the learned-policy viewer after training:

```sh
make view-learned
```

The learned viewer expects `runs/stage2-seed0/latest.pt`. Use `make help` to list all available commands. The main training flow is:

```sh
make train-smoke
make eval-smoke
make train-stage1
make eval-stage1
make train-stage2
make eval-stage2
```

The smoke run checks the training and checkpoint pipeline; it is not evidence that the learned policy has solved the task. Stage 1 trains on hold/follow scenarios, and stage 2 resumes from stage 1 with crossing and head-on disturbances. Multi-seed repeats use `make train-stage1-multi` / `eval-stage1-multi` / `train-stage2-multi` / `eval-stage2-multi`. Extra eval targets: `eval-heldout` (seeds 100000+), `eval-noisy`, `eval-dense`. See [sofar.md](sofar.md) for the current research conclusions, setup summary and paper-to-implementation mapping.

## Run the interactive laboratory

On this Mac, double-click **[Launch SwarmFlow.command](Launch%20SwarmFlow.command)**. It opens the browser and starts the server, or reuses it if already running. Click **Repeat flights** for a continuous demonstration. Keep the launcher's Terminal window open; Control+C stops a server it started. Local demos work without internet after the Python environment is installed.

The launcher works from this folder even if it is moved. On a new Mac, install [uv](https://docs.astral.sh/uv/getting-started/installation/) first; the launcher will create the Python environment on first use. That initial setup requires internet. Logs are saved to `runs/local-demo.log`.

The launcher always selects the **predictive baseline**. It reuses only a predictive viewer and skips ports occupied by learned-policy viewers. If flight behavior changes after following the RL training guide, check **Active controller**: `learned checkpoint` is the experimental neural policy. Reopen the launcher for the earlier predictive demonstration. Both modes run locally; the controller determines their behavior.

From this `neww` directory:

```sh
uv sync --python 3.12 --extra dev --cache-dir .uv-cache
.venv/bin/python -m swarmflow view
```

Open **http://127.0.0.1:8765**. Choose a scenario and press **Start flight**. During a flight, press **Send a crossing object** or **Send from ahead**. Adjust object speed/radius, drag to orbit, switch to top view, and save a PNG. Change scenario, drone count or seed, then press **Reset**. The default episode is 24 simulated seconds; use `view --duration 60` for a longer interactive session.

When a test finishes or fails, click **Run again** to reset and start the selected scenario in one click. **Repeat flights** starts flying immediately and automatically starts another episode whenever the current test ends. **Pause flight** stops the motion and turns repetition off. **Reset** resets the scene without starting it.

The default viewer runs a **predictive classical baseline**. It is immediately usable and supplies a reference for evaluating RL. A learned policy is shown only when you explicitly load a checkpoint. Failure ends the flight; there is no hidden position correction or collision-avoidance wrapper around a learned policy.

## What is implemented

- PettingZoo parallel environment with Gymnasium spaces and simultaneous 3D velocity commands.
- Fixed target-relative slot identity, finite controller response, acceleration/speed command limits, and collision checks over every integration segment.
- Local masked neighbor/obstacle observations, configurable sensor range, position noise and observation dropout.
- Nine scenarios: hold, follow, crossing, head-on, repeated crossings, static spheres, a dense sphere cluster, smooth target turns, and wind gusts.
- Live object injection, 3D browser visualization, individual slot markers, trails and telemetry.
- A sampled predictive baseline, a no-avoidance ablation, and a shared neural policy with masked set encoders.
- Behavior-cloning warm start, clipped independent PPO, checkpoint save/reload/resume, and seeded JSON evaluation.
- Per-episode collision/failure, minimum surface clearance, slot and centroid RMSE, command smoothness, and per-event recovery measurements. Evaluation reports include success rate and slot-RMSE confidence intervals; `python -m swarmflow compare a.json b.json` diffs two reports.

This first environment models **stabilized translational flight**, not full rotor/attitude physics. Drone tilt in the viewer is illustrative. There is no downwash, motor model, camera perception, physical contact response or hardware flight validation. The plan specifies a later PyBullet validation stage.

## Research conclusions

The papers led to a scoped benchmark rather than a claim of a new flight-control method. The evidence supports shared local policies, permutation-invariant neighbor representations, PPO as a reasonable cooperative-learning baseline, predictive relative-motion reasoning, and explicit formation-recovery measurements. It does not support claiming formal safety, dense-obstacle generalization, physical-flight validity or a qualified learned policy from the current development runs.

Accordingly, the project first learns velocity-level swarm movement above a stabilized flight model, keeps persistent target-relative slot identities, compares against a visible predictive controller, and measures collision, clearance, formation error and recovery separately. Full multirotor physics and hardware validation remain future experiments. Dense sphere clusters and optional rangefinder hits are in the environment; full-height mesh walls are not. The complete paper-by-paper rationale and implementation status are documented in [sofar.md](sofar.md).

## Evaluate and test

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m swarmflow eval --seeds 3 --output runs/baseline.json
.venv/bin/python -m swarmflow eval --policy slot --scenarios crossing head_on double_crossing --seeds 3 --output runs/no-avoidance.json
.venv/bin/python -m swarmflow eval --drones 16 --scenarios crossing head_on --seeds 2 --seed-start 2000 --output runs/scaling-16.json
```

Use `--trace` to save trajectory snapshots alongside the JSON report. Seeds 1000–1999 are development validation seeds; 2000–2001 have also been used for the included scaling check. Reserve seeds **100000+** for an untouched final test set. Training episode generation uses 0–999; the CLI training seed controls network initialization, task selection and episode sampling.

Initial measured results: **24/24 nominal eight-drone baseline episodes completed without collision**, with all 12 encountered recovery events completed; the no-avoidance controller collided in 9/9 obstacle trials. Four additional 16-drone trials also completed without collision. These small development checks do not establish a general collision-free guarantee. See [validation details and limitations](docs/validation.md).

## Train RL

For a presentation walkthrough covering what is already saved, live training output, evaluation, learned-policy viewing, and resuming progress, see [Showing how SwarmFlow learns](docs/training-demo.md).

First verify the pipeline with a small budget and a new output directory:

```sh
.venv/bin/python -m swarmflow train --steps 512 --horizon 64 --epochs 2 --clone-steps 128 --clone-epochs 3 --scenarios hold follow crossing --output runs/my-smoke
.venv/bin/python -m swarmflow eval --policy learned --checkpoint runs/my-smoke/latest.pt --scenarios follow crossing --seeds 3 --output runs/my-smoke/evaluation.json
```

A smoke checkpoint tests the machinery, not the task. The included `runs/smoke-policy/latest.pt` is **not a qualified flight policy**. Its performance report is separate from the baseline report.

For the first actual experiment, start with hold/follow, evaluate, then introduce crossing events while retaining simpler tasks:

```sh
.venv/bin/python -m swarmflow train --steps 50000 --clone-steps 2000 --scenarios hold follow --seed 0 --output runs/stage1-seed0
.venv/bin/python -m swarmflow eval --policy learned --checkpoint runs/stage1-seed0/latest.pt --scenarios hold follow --seeds 20 --output runs/stage1-seed0/validation.json
.venv/bin/python -m swarmflow train --resume runs/stage1-seed0/latest.pt --steps 100000 --scenarios hold follow crossing head_on --seed 0 --output runs/stage2-seed0
.venv/bin/python -m swarmflow eval --policy learned --checkpoint runs/stage2-seed0/latest.pt --seeds 20 --output runs/stage2-seed0/validation.json
.venv/bin/python -m swarmflow view --checkpoint runs/stage2-seed0/latest.pt
```

Budgets above are starting experiments, not convergence promises. Do not advance a stage merely because it used its step budget. Repeat with at least three independent training seeds and compare success, recovery and clearance together. PPO's reported `steps` count environment advances; multiply by drone count for agent transitions. Behavior-cloning samples are counted separately in the log. Default training is **parameter-sharing IPPO with a local critic**; use `--critic central` for a MAPPO-style value head and `--failure-reset` for pre-collision restores. Multi-seed stages: `make train-stage1-multi` / `train-stage2-multi`. Resume restores network and optimizer, then starts a fresh seeded episode; it is not exact trajectory continuation.

## Files

- `src/swarmflow/env.py`, `scenarios.py`, `config.py`: environment, scenarios and experiment contract.
- `src/swarmflow/baseline.py`: predictive and slot-only controllers.
- `src/swarmflow/policy.py`, `train.py`, `evaluate.py`: learning and measurement.
- `src/swarmflow/server.py`, `web/`: local interactive viewer; no JavaScript build system or external CDN required.
- `tests/`: behavior and algorithm regression checks.
- `docs/research-plan.md`, `docs/implementation.md`, `docs/validation.md`: sources, specification, and evidence.
- `sofar.md`: current research conclusions and paper-to-feature implementation map.
- `runs/`: generated results and checkpoints, ignored by Git.

The environment and all project artifacts live here in `neww`. No code or assets are imported from the earlier project or the reference repository.
