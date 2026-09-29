# Configuration and reproducible execution

[Implementation sequence](README.md) · [Project equations](project_context.md) · [Paper index](README.md#paper-index)

These are defaults from the current code, not universal constants from research papers. `Config` describes the environment; `train()` has generic defaults; `train --stage N` and `pipeline` override those with curriculum settings. Saved run `config.json` is authoritative for that run.

## Environment defaults

Source: [config.py](../src/swarmflow/config.py).

- `drones = 8` — Team size, allowed 2–32.
- `dt = 0.05` — Control interval in seconds.
- `substeps = 5` — Integration substeps per control step.
- `duration = 24.0` — Episode time limit in seconds.
- `radius = 0.18` — Drone collision-sphere radius in metres.
- `spacing = 1.25` — Requested formation spacing in metres.
- `altitude = 3.0` — Nominal target/formation altitude in metres.
- `stand_off = 3.0` — World-x stand-off behind the target in metres.
- `max_speed = 3.0` — Desired-velocity norm limit in m/s.
- `max_accel = 5.0` — Commanded-acceleration norm limit in m/s² before wind.
- `response_time = 0.28` — First-order velocity response time constant in seconds.
- `target_speed = 0.55` — Nominal target x velocity in m/s.
- `sensor_range = 7.0` — Visible-object center-distance limit in metres.
- `neighbors = 6` — Maximum real neighbor rows.
- `obstacles = 6` — Maximum obstacle rows, including optional ray hits.
- `noise = 0.0` — Gaussian relative-position noise standard deviation in metres.
- `dropout = 0.0` — Independent real-row dropout probability.
- `recovery_position = 0.35` — Maximum per-drone slot error in metres for settling.
- `recovery_velocity = 0.35` — Maximum per-drone target-relative speed error in m/s for settling.
- `recovery_hold = 1.0` — Required continuous settling duration in seconds.
- `rays = 0` — Horizontal rangefinder directions; zero disables rays.
- `ray_range = 5.0` — Ray hit range cap in metres.
- `geometry = 'grid'` — grid, circle, square or triangle.
- `world = 'empty'` — empty, hills, canyon or pillars.

All numeric fields must be finite. Counts are positive integers (ray count can be zero); drone count is 2–32; physical positive parameters must be >0; spacing must exceed `2*radius+0.2`; dropout is in [0,1), noise ≥0; target speed is in [0,max_speed); dt ≤0.1 and dt/substeps ≤response_time/2. Config is a frozen dataclass. Geometry/world names are validated. CLI `eval`/`train` overrides expose a subset; Python configuration can change more fields.

## Training defaults

Source: [train.train](../src/swarmflow/train.py). Required inputs are `config` and `scenarios`.

- `total_steps = 10000` — Additional team PPO steps unless continuing the original budget.
- `horizon = 128` — Rollout ticks per environment.
- `epochs = 4` — Maximum PPO epochs per update.
- `seed = 0` — Training RNG and initialization seed.
- `output = 'runs/train'` — Run folder.
- `clone_steps = 0` — Team steps per cloning/DAgger round; zero skips cloning.
- `clone_epochs = 15` — Passes over the growing teacher dataset per round.
- `resume = None` — Checkpoint path for initialization.
- `device = 'cpu'` — cpu, mps or cuda.
- `critic = 'local'` — local or central value input.
- `failure_reset = False` — Enable limited collision-state restores.
- `envs = 8` — Number of independent env instances with batched inference.
- `separate_critic = True` — Independent critic encoder; false shares actor features.
- `lr = 0.0003` — Initial PPO Adam rate.
- `minibatch = 1024` — Agent samples per PPO minibatch, with whole-team grouping for central critic.
- `gamma = 0.99` — Discount factor.
- `lam = 0.95` — GAE lambda.
- `clip = 0.2` — PPO likelihood-ratio clip width.
- `entropy = 0.0` — Latent Gaussian entropy coefficient.
- `target_kl = 0.03` — Epoch-level approximate KL threshold.
- `dagger_rounds = 0` — Extra student-driven rounds after noisy teacher round zero.
- `clone_noise = 0.3` — Round-zero executed-action Gaussian std.
- `init_log_std = None` — Derived as −1.6 with cloning, otherwise −1, when None.
- `critic_warmup = None` — Derived as 10 with cloning, 5 when resuming a different folder without cloning, otherwise 0.
- `normalize_rewards = True` — Enable running discounted-return std reward scaling.
- `eval_every = 25000` — Team steps between selection evaluations; 0 disables periodic eval.
- `eval_seeds = 5` — Number of selection episode seeds per scenario, starting at 5000.
- `eval_scenarios = None` — Defaults to training scenarios.
- `snapshot_every = 0` — Periodic retained checkpoint steps; 0 disables snapshots.
- `threads = None` — Default min(4, CPU count) PyTorch threads.
- `continue_run = False` — Resume original output-folder budget/settings.
- `stage = None` — Stage label in metadata.
- `width = 64` — Encoder hidden width.
- `min_log_std = -3.0` — Latent Gaussian lower log-std clamp.
- `bc_coef = 1.0` — Imitation coefficient at stage start.
- `bc_coef_end = 0.5` — Imitation coefficient at stage end.
- `bc_batch = 1024` — Teacher samples per PPO minibatch.
- `pipeline_version = None` — Metadata identifying recipe version.
- `teacher_margins = (0.7, 0.3)` — Soft/hard prediction penalties for teacher only, in metres.
- `hard_weight = 1.0` — Sampling weight for near-object/near-neighbor demonstration states.
- `gap_features = False` — Append explicit surface gap/proximity/closing speed to set rows.

ActorCritic construction outside the trainer has older-compatible defaults (`separate_critic=False`, `min_log_std=-5`, width 64). The trainer deliberately overrides them. `clone()` as a helper defaults to teacher margins 0.35/0.08; the trainer passes its 0.7/0.3 margins explicitly. Do not confuse helper defaults with the executed curriculum.

## Curriculum stages

Pipeline version **4**, source [curriculum.py](../src/swarmflow/curriculum.py).

1. **Stage 1:** hold/follow/turn; 1,000,000 PPO team steps; clone 10,000 team steps per round; three extra DAgger rounds; PPO rate 3e-4; failure resets off; fresh model.
2. **Stage 2:** retain stage 1 and add crossing/head_on; 1,000,000 PPO team steps; clone 20,000 per round; three extra DAgger rounds; PPO rate 1e-4; failure resets on; start from stage 1 best.pt.
3. **Stage 3:** retain stage 2 and add double_crossing/gust; 1,000,000 PPO team steps; clone 20,000 per round; three extra DAgger rounds; PPO rate 1e-4; failure resets on; start from stage 2 best.pt.

Common overrides: 8 environments; horizon 256; five epochs; minibatch 2048; selection every 100,000 steps with ten seeds; snapshots every 500,000 steps; width 128; log-std floor −3; entropy 0; imitation coefficient 1.0 →0.5; teacher margins 0.7/0.3 m; hard weight 1.0; gap features off. Other generic defaults still apply, including 15 clone epochs, local separate critic, gamma .99, lambda .95 and KL .03.

A full default update has 2,048 team steps / 16,384 drone transitions. Nominal full-pipeline PPO total is 3 M team steps /24 M drone transitions, plus 200,000 cloning team steps /1.6 M labels over the three stages. Intermediate best-checkpoint lineage counts can be lower than total work already performed. Wall-clock estimates in old documents are hardware/run measurements, not guarantees.

## Commands and entry points

Run from the repository root after setup:

```sh
make install
make test
make training
make run train
make train-full-multi
make train-stage2
make continue-stage2
make eval-stage2
make showcase
make run
make view-baseline
make tests
make test-list
make eval-noisy
make eval-dense
```

These commands are documented for future use; they were not run to create this context folder. `make training` runs `python -m swarmflow pipeline`; `make run train` adds the live viewer. Multi-seed defaults are 0 1 2. `make run` starts a viewer and does not train. The macOS launcher bootstraps the environment if needed, selects/reuses a local server and writes `runs/local-demo.log`. `caffeinate -i` is optional macOS sleep prevention during long jobs.

The CLI exposes view, eval, compare, test, train, pipeline and showcase. `__main__.py` calls `cli.main`; the installed `swarmflow` script uses the same function. For changed settings use `python -m swarmflow <command> --help`; gamma/lambda, for example, are Python training parameters rather than exposed CLI flags in this revision. Numbered suite tests 9–11 may train a stage if its checkpoint is missing. `make train-smoke` is a machinery check with tiny budgets, not a qualified learned controller.

`--resume` means initialize another run with a compatible checkpoint; `--continue` means finish the original saved budget. Existing output checkpoints are protected against accidental fresh overwrite. Pipeline orchestration archives outdated version folders and skips completed stages. Starting a new showcase archives the previous showcase. Explicit checkpoint viewing does not convert old formats to trainable current formats.

## Dependencies and build

Source: [pyproject.toml](../pyproject.toml), [uv.lock](../uv.lock), [Makefile](../Makefile).

- Python >=3.11,<3.14; setup selects Python 3.12.
- NumPy >=1.26,<3: vector simulation, geometry, seeded randomization and array metrics.
- Gymnasium >=1.0,<2: observation/action spaces.
- PettingZoo >=1.24,<2: simultaneous multi-agent ParallelEnv contract.
- PyTorch >=2.2,<3: MLP/attention/Gaussian policy, Adam, autograd and checkpoints.
- pytest >=8,<9 (dev extra): unit/integration tests.
- setuptools >=68: package build backend and packaged web assets.
- uv: environment/lockfile workflow; not an imported simulator module.
- Standard library: HTTP server, threading/subprocesses, paths/JSON, dataclasses, argument parsing, timing and queues.
- Browser HTML/CSS/JavaScript/Canvas2D: rendering and local JSON client; no external frontend dependencies.

`uv.lock` resolves exact package versions and transitive dependencies for supported environments; the lockfile is the complete dependency record. `.venv`, `.uv-cache`, `.pytest_cache`, bytecode and egg-info are generated tooling, not additional project research. `runs/` is **not** ignored in the current .gitignore (its ignore line is commented out).
