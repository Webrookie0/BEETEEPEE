# SwarmFlow

An independent reinforcement-learning environment for a drone swarm that follows a reference target, makes room for obstacles, and recovers the **same assigned formation slots** afterward.

Start with the [research and execution plan](docs/research-plan.md). It connects six primary research sources to concrete design decisions and distinguishes established results from hypotheses. The main recommendation is to learn swarm motion above a stable flight controller and study formation recovery explicitly.

## Setup and run

SwarmFlow runs locally on Python 3.12. From the project directory, install the environment and development dependencies:

```sh
make install        # uv sync --python 3.12 --extra dev --cache-dir .uv-cache
```

Start the interactive viewer with:

```sh
make run
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765). The viewer automatically loads the trained model in `models/showcase/best.pt` when it exists; otherwise it runs the predictive classical baseline. The **Controller** menu switches between the learned policy, the baseline and the slot-only ablation.

**To train the model yourself, follow [guide.md](guide.md)**; the whole curriculum is one command:

```sh
caffeinate -i make training        # stage 1 -> 2 -> 3 (about 3.5 h), then the held-out showcase test
caffeinate -i make run train       # the same, with the viewer open: live progress, then the best model
make run                           # viewer with the best model and its "Did the model learn correctly?" panel
```

[training.md](training.md) explains the training method, the code, and why the earlier 2–5 minute runs did not learn. Use `make help` to list all commands. Extra baseline eval targets: `eval-heldout` (seeds 100000+), `eval-noisy`, `eval-dense`.

## Run the interactive laboratory

On this Mac, double-click **[Launch SwarmFlow.command](Launch%20SwarmFlow.command)**. It opens the browser and starts the server, or reuses it if already running. Click **Repeat flights** for a continuous demonstration. Keep the launcher's Terminal window open; Control+C stops a server it started. Local demos work without internet after the Python environment is installed.

The launcher works from this folder even if it is moved. On a new Mac, install [uv](https://docs.astral.sh/uv/getting-started/installation/) first; the launcher will create the Python environment on first use. That initial setup requires internet. Logs are saved to `runs/local-demo.log`.

The launcher starts the same viewer as `make run`: with a trained model under `models/` it opens in **learned checkpoint** mode, otherwise in **predictive baseline** mode. It reuses a SwarmFlow viewer that is already running. Check **Active controller** to see which one is flying, and use the **Controller** menu to switch.

From this `neww` directory:

```sh
uv sync --python 3.12 --extra dev --cache-dir .uv-cache
.venv/bin/python -m swarmflow view
```

Open **http://127.0.0.1:8765**. Choose a scenario and press **Start flight**. During a flight, press **Send a crossing object** or **Send from ahead**. Adjust object speed/radius, drag to orbit, switch to top view, and save a PNG. Change scenario, drone count or seed, then press **Reset**. The default episode is 24 simulated seconds; use `view --duration 60` for a longer interactive session.

When a test finishes or fails, click **Run again** to reset and start the selected scenario in one click. **Repeat flights** starts flying immediately and automatically starts another episode whenever the current test ends. **Pause flight** stops the motion and turns repetition off. **Reset** resets the scene without starting it.

The **predictive classical baseline** is always available as the reference for evaluating RL. The learned policy is the trained network from `models/showcase/best.pt` (or `--checkpoint`). Failure ends the flight; there is no hidden position correction or collision-avoidance wrapper around a learned policy.

## What is implemented

- PettingZoo parallel environment with Gymnasium spaces and simultaneous 3D velocity commands.
- Fixed target-relative slot identity, finite controller response, acceleration/speed command limits, and collision checks over every integration segment.
- Local masked neighbor/obstacle observations, configurable sensor range, position noise and observation dropout.
- Nine scenarios: hold, follow, crossing, head-on, repeated crossings, static spheres, a dense sphere cluster, smooth target turns, and wind gusts.
- Live object injection, 3D browser visualization, individual slot markers, trails and telemetry.
- A sampled predictive baseline, a no-avoidance ablation, and a shared neural policy with masked set encoders.
- Behavior-cloning warm start with DAgger, clipped parameter-sharing PPO with a separate critic, 8 parallel environments, reward scaling, a three-stage curriculum, best-checkpoint selection on validation seeds, resumable training, and a held-out showcase test. See [training.md](training.md).
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

Step-by-step commands are in **[guide.md](guide.md)**; the method, the code walkthrough and the research behind each choice are in **[training.md](training.md)**.

```sh
make test                          # unit tests
make train-smoke                   # about 1 min; checks the machinery only, not the task
caffeinate -i make training        # full curriculum for seed 0, then models/showcase (re-run to continue after an interruption)
caffeinate -i make run train       # the same with the live viewer (python -m swarmflow pipeline --view)
make train-full-multi              # seeds 0 1 2; the showcase keeps the best seed
make train-stage1                  # or one stage at a time (train-stage2, train-stage3, continue-stageN, eval-stageN)
make showcase                      # choose the best run, test it on held-out seeds, store it in models/showcase
```

Each stage writes `models/stageN-seedS/` with `best.pt` (best validated checkpoint), `latest.pt` (for continuing), logs and settings; the next stage starts from the previous stage's `best.pt`. PPO's reported `steps` count environment advances summed over the 8 parallel environments; multiply by the drone count for agent transitions. Default training is **parameter-sharing IPPO with a separate local critic**; use `--critic central` for a MAPPO-style value head and `--failure-reset` for pre-collision restores. Checkpoints from before this pipeline (`swarmflow-v1/v2` in `runs/`) can still be viewed but not resumed.

## Files

- `src/swarmflow/env.py`, `scenarios.py`, `config.py`: environment, scenarios and experiment contract.
- `src/swarmflow/baseline.py`: predictive and slot-only controllers.
- `src/swarmflow/policy.py`, `train.py`, `evaluate.py`: learning and measurement.
- `src/swarmflow/curriculum.py`: training stages, the resumable pipeline, and the showcase (best model + held-out test).
- `src/swarmflow/server.py`, `web/`: local interactive viewer; no JavaScript build system or external CDN required.
- `tests/`: behavior and algorithm regression checks.
- `guide.md`, `training.md`: how to train, and how the training works.
- `docs/research-plan.md`, `docs/implementation.md`, `docs/validation.md`: sources, specification, and evidence.
- `sofar.md`: current research conclusions and paper-to-feature implementation map.
- `models/`: trained stages (`stageN-seedS/`) and the selected model (`showcase/`).
- `runs/`: evaluation reports, test-suite outputs and older checkpoints.

The environment and all project artifacts live here in `neww`. No code or assets are imported from the earlier project or the reference repository.
