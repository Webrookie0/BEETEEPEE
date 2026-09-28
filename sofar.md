# SwarmFlow so far

This document records the current setup, execution flow, research conclusions, what has been implemented from each paper, the training runs so far and the current status. It is a project status document, not a claim that the benchmark or learned policy is publication-ready. For the complete picture in one file (timeline, training phases with diagrams, every run, every problem and a file map), read [knowledge.md](knowledge.md).

Last updated: 28 September 2026 (pipeline version 4 committed; its full training run not done yet).

## Current status in short

- **Goal**: reinforcement learning for drone swarms. One shared, decentralized policy keeps 8 drones in formation behind a moving target, avoids moving objects and returns every drone to its own slot.
- **Best result so far: run 3 (pipeline v3)**. On the held-out test (7 scenarios × 20 unseen seeds) it scores 0.936 vs 0.970 for the hand-coded baseline. 138/140 episodes had no failure; the 2 failures were head-on grazes. Slot precision passes (0.023 / 0.027 / 0.045 m). Verdict: **not learned yet**, because the safety check needs zero failures.
- **Pipeline v4** (`c4f7249`) targets those last failures: a more cautious teacher (0.7 / 0.3 m), recovery measured after overlapping objects, and 1M PPO steps in stages 2-3. All 40 unit tests pass and a tiny end-to-end run finishes; **the full run (about 3-3.5 h) is the next step**.
- **Safety-filter experiment** (`runs/experiments/shield.py`): with the teacher taking over only when a drone is predicted within 0.1 m of contact in the next 1 s, run 3's model has 0/140 failures on development seeds and passes every check. The teacher is used on 0.01% of drone-steps. Not part of the pipeline yet.
- The viewer's `models/showcase/` still holds **run 2's** model; run 3's is in `models/previous-runs/run3-pipeline-v3/`.

## Run setup

SwarmFlow is a local Python project using Python 3.12, NumPy, Gymnasium, PettingZoo and PyTorch. The included `Makefile` is the recommended entry point.

```sh
cd BTP/Drone-Swarm
make install
```

This creates or updates `.venv` using `uv`. To run the browser laboratory:

```sh
make run
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The viewer loads the best trained model (`models/showcase/best.pt`, otherwise the best stage model under `models/`) and falls back to the predictive baseline when none exists. `make view-baseline` always shows the baseline.

Run the tests and baseline evaluation with:

```sh
make test     # 40 unit tests
make eval     # predictive baseline -> runs/baseline.json
make tests    # menu of the 11 numbered validation tests
```

## Training flow

One command runs the whole pipeline for seed 0 and can be re-run to continue after an interruption:

```sh
caffeinate -i make training     # or: caffeinate -i make run train  (with the live viewer)
```

It runs three curriculum stages, then the showcase:

| Stage | Scenarios | Cloning (teacher 0.7 / 0.3 m) | PPO steps (v4) | Starts from |
| --- | --- | --- | --- | --- |
| 1 | hold, follow, turn | DART + 3 DAgger rounds, 4 × 10,000 steps | 1,000,000 (lr 3e-4) | random weights |
| 2 | + crossing, head_on | 4 × 20,000 steps | 1,000,000 (lr 1e-4, crash replay) | stage 1 `best.pt` |
| 3 | + double_crossing, gust | 4 × 20,000 steps | 1,000,000 (lr 1e-4, crash replay) | stage 2 `best.pt` |
| Showcase | all 7 scenarios | – | – | the best stage 3 model; tested on seeds 100000-100019 against the baseline |

Each stage runs:

1. cloning: DART noise in round 0, DAgger in rounds 1-3, 15 epochs per round;
2. validation of the cloned start;
3. critic-only warm-up;
4. PPO: 8 environments × 256 steps, 5 epochs, minibatch 2,048, imitation loss weight 1.0 → 0.5, KL limit 0.03;
5. validation every 100,000 steps on selection seeds 5000+, keeping the best as `best.pt`.

Results go to `models/stageN-seed0/` and `models/showcase/` (`SUMMARY.md` gives the verdict). Other commands:

- more seeds: `make train-full-multi` (`SEEDS` default `0 1 2`);
- a single stage: `make train-stage1/2/3`, then `make eval-stage1/2/3`;
- only the machinery check: `make train-smoke`;
- extra baseline checks: `make eval-heldout`, `make eval-noisy`, `make eval-dense`.

Details: [guide.md](guide.md) and [training.md](training.md).

## How the paper conclusions were reached

The project began by reading six primary sources or official project pages and translating each source into either a design decision, a comparison baseline, a limitation or a future validation step. The conclusions were deliberately kept narrower than the papers' full claims:

1. Reuse ideas that are directly relevant to decentralized cooperative flight: shared policies, local relative observations, permutation-invariant neighbor processing and explicit formation measurements.
2. Separate stabilized swarm movement from low-level motor and attitude control so failures can be attributed to coordination and avoidance rather than an unmanageable combined optimization problem.
3. Treat moving spherical objects as the first benchmark scope. Dense static walls and corridors require richer distance or ray observations and should not be inferred from this task.
4. Use PPO as a credible learning baseline. Default training is parameter-sharing IPPO with a local critic; `--critic central` is an optional MAPPO-style comparison, not a claim of full MAPPO.
5. Use predictive relative-motion reasoning in the classical controller, while avoiding the claim that the controller is ORCA or has ORCA's formal guarantees.
6. Keep PyBullet and hardware validation as later stages. The current model is a reduced-order stabilized-flight benchmark, not a physical multirotor simulator.

These conclusions are hypotheses and engineering choices supported by the reviewed literature, not conclusions that the training runs so far prove. The full source notes are in [docs/research-plan.md](docs/research-plan.md). The training-method papers (PPO, GAE, DAgger, DART, DAPG and others) are listed with the code they changed in [training.md](training.md) section 7 and [knowledge.md](knowledge.md) section 5.

## Paper-to-implementation map

### Batra et al. — Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning, CoRL 2021

The paper supports shared decentralized policies, local neighbor position/velocity information, permutation-aware set processing, PPO training and explicit formation re-establishment around moving obstacles. It also shows that the general behavior already has precedent, so SwarmFlow does not present the basic task as an entirely new idea.

Implemented in SwarmFlow:

- Shared neural policy weights across drones.
- Local self, neighbor and obstacle observations using relative position and velocity.
- Masked attention set encoders, making neighbor order irrelevant.
- PPO training after a behavior-cloning + DAgger warm start in every stage.
- Moving spherical obstacles and explicit recovery to each drone's original slot.
- Separate collision, formation-error and recovery measurements.
- Since v2: the crash penalty goes to the drones involved (not the whole team) and a smooth obstacle warning zone starts at 1 m.

Not copied or claimed: the paper's direct-thrust setup, very large training budget, full reported performance and any claim that this implementation reproduces its results.

### Huang et al. — Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep Reinforcement Learning, ICRA 2024

This work motivated attention to obstacle representation, collision-state resets and the limitations of nearest-obstacle-only sensing. Its dense-obstacle setting is materially different from the current moving-object benchmark, so it led to a scope boundary: dense walls and corridors require a later sensing and navigation experiment.

Implemented or reflected:

- Bounded local obstacle observations with visibility masks and configurable sensor capacity.
- Optional observation noise and dropout for robustness experiments.
- Optional short-range rays folded into leftover obstacle rows (default off).
- A `dense` scenario of 8–12 static spheres (not full-height mesh walls).
- Fresh on-policy PPO rollouts after cloning; stale transitions are not reused across resets.
- Swept collision detection and terminal failure metrics.
- Masked multi-head attention set encoders for neighbors and obstacles.
- `--failure-reset` (restore a state about 2 s before a crash, fresh on-policy rollouts), **on in stages 2-3 since v2**; `--critic central` (MAPPO-style shared value) optional.
- Gap (distance) features in the encoder, tested in v4 and left off (5 failures instead of 1 on development seeds).

Not yet implemented: the paper's dense full-height static-obstacle benchmark and its full training stack.

### Huang et al. — QuadSwarm: A Modular Multi-Quadrotor Simulator for Deep Reinforcement Learning with Direct Thrust Control, ICRA 2023 workshop

The paper and repository supported separating simulation, rendering, scenario generation and randomization. They are treated as references, not as the base code or as evidence of robustness. Its training code was also the reference for the v2 fine-tuning settings (learning rate 1e-4, wider network, no entropy bonus).

Implemented in SwarmFlow:

- A standalone environment separated from the browser renderer.
- Reproducible seeded scenario generation, including randomized event times and headings.
- Configurable drone counts, formation geometries (grid, circle, square, triangle), worlds (empty, hills, canyon, pillars), scenarios, sensor limits and disturbances.
- A local web viewer that consumes simulation state through a Python server.

Not implemented: direct thrust dynamics or the QuadSwarm simulator itself.

### Yu et al. — The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games, NeurIPS 2022 Datasets and Benchmarks

This work supports treating cooperative PPO as a serious baseline, while not implying that game benchmarks establish drone performance. It informed the initial learning choice and the conservative naming of the algorithm.

Implemented in SwarmFlow:

- Parameter-sharing independent PPO for simultaneous drone actions (default local critic, separate from the actor).
- Optional centralized critic over mean-pooled agent embeddings (`--critic central`); not a claim of MAPPO parity.
- GAE, advantage normalization, reward scaling, ratio clipping, gradient clipping and approximate-KL early stopping.
- 8 parallel environments, 16,384 samples per update, 5 epochs, minibatch 2,048.
- Checkpointing, resume support and seeded evaluation with success-rate and slot-RMSE confidence intervals.

Not implemented: distributed-scale data collection or any claim of parity with the paper's environments.

### Panerati et al. — Learning to Fly, IROS 2021

This paper provides a route toward full multirotor physics, aerodynamic effects and controller comparisons. It led to the decision to keep the current benchmark fast and interpretable while reserving PyBullet validation for a separate stage.

Implemented in SwarmFlow:

- A documented interface between task logic, controller actions and simulation state.
- Velocity-level commands with finite response, acceleration limits and fixed control frequency.
- Validation notes that explicitly distinguish stabilized translational flight from rotor physics.

Not implemented: six-degree-of-freedom dynamics, thrust/tilt limits, downwash, aerodynamic effects, sim-to-real calibration or hardware flight.

### van den Berg et al. — Reciprocal n-body Collision Avoidance, ISRR 2011

The ORCA work motivated predicting relative motion and dividing avoidance responsibility among agents. The project explicitly avoids calling its controller ORCA: it does not solve ORCA's linear program and has no formal safety guarantee. External moving objects are treated as non-cooperative.

Implemented in SwarmFlow:

- A sampled receding-horizon predictive baseline, which is also the imitation teacher (with wider margins: 0.6 / 0.25 m in v3, 0.7 / 0.3 m in v4).
- Relative position and velocity in local observations.
- Predicted finite-response motion over a short horizon.
- Clearance penalties, detours and waiting behavior in the baseline.
- Continuous swept collision checks over integration segments.

Not implemented: ORCA's reciprocal velocity-obstacle formulation, linear program or formal guarantees.

### Papers added while fixing the training runs

- **Schmitt et al. 2018 (Kickstarting)** and **Rajeswaran et al. 2018 (DAPG)**: imitation loss with a decaying weight inside PPO (v2).
- **Mayne et al. 2005 (robust MPC, constraint tightening)**: the teacher keeps a larger margin than required, so the student's imitation error still leaves clearance (v3/v4).
- **Williams et al. 2017 (MPPI)**: tested a temperature-averaged teacher; rejected because recovery got slower (v4, option kept but off).
- **Wołczyk et al. 2024 (fine-tuning as forgetting)**: explains why PPO did not improve on the cloned start in stages 2-3 (v4 shortens those stages).

## What is implemented overall

- PettingZoo parallel environment with Gymnasium spaces.
- Immutable target-relative formation slots in four geometries.
- Stabilized 3D translational dynamics with finite response and acceleration limits.
- Swept drone/drone and drone/object collision checks.
- Eleven scripted scenarios: hold, follow, crossing, head_on, double_crossing, slalom, dense, turn, gust, pilot, terrain_follow. The training/test benchmark uses seven: hold, follow, crossing, head_on, double_crossing, turn, gust.
- Four worlds: empty (used for training), hills, canyon, pillars.
- Local masked observations with optional noise, dropout, and optional ray hits in leftover obstacle slots.
- Predictive baseline, slot-only no-avoidance ablation and shared learned policy (attention encoders, separate critic, width 128).
- Bounded reward: +1 per second alive minus capped costs, −100 for each drone involved in a failure.
- Resumable three-stage pipeline: cloning + DAgger, critic warm-up, PPO with imitation loss, crash replay, best-checkpoint selection on dedicated seeds, and a held-out showcase with three acceptance checks.
- Separate seed ranges: training 0-999, selection 5000+, development 1000+, final test 100000+.
- Browser viewer with live injections, trails, slot markers, telemetry, PNG export, live training progress and a "Did the model learn correctly?" panel.
- Metrics for failure, collision, clearance, slot RMSE, centroid RMSE, smoothness and event recovery (measured after overlapping objects since v4), plus success rate and slot-RMSE CI; `swarmflow compare` diffs two reports.
- Numbered validation suite (`make tests`), multi-seed training, held-out, noisy and dense evaluations.

## Training runs so far

Held-out test on seeds 100000-100019 (details in [models/previous-runs/README.md](models/previous-runs/README.md)):

| | Run 1 (v1) | Run 2 (v2) | Run 3 (v3) | Baseline |
| --- | --- | --- | --- | --- |
| Episodes without failure | 138/160 | 148/160 | 138/140 | 160/160 (run 3 seeds: 140/140) |
| Recovered within 5 s | 38/80 | 66/80 | 70/80 (78/80 with the v4 rule) | 76/80 (80/80 with the v4 rule) |
| Test score | 0.513 | 0.815 | 0.936 | 0.962 (run 3 seeds: 0.970) |
| Slot RMSE hold / follow / turn | 0.15 / 0.19 / 0.19 m | 0.08 / 0.05 / 0.08 m | 0.023 / 0.027 / 0.045 m | 0.014 / 0.014 / 0.033 m |

What went wrong in each run and what changed: `training.md` Parts C (run 1 → v2), D (run 2 → v3) and E (run 3 → v4).

## Current limitations

- Hardware flight, full 6DOF/PyBullet rotor physics and ORCA's linear program are not implemented.
- No trained model has passed all three acceptance checks yet. Pipeline v4 has not been trained in full.
- Only seed 0 has been trained; the same recipe varies by 1-3 failures between runs, so multi-seed results are still needed.
- Training covers the empty world, 8 drones and 7 scenarios. Slalom, dense spheres, other worlds, geometries and swarm sizes are untested for the learned policy.
- In stages 2-3, PPO has not improved on the cloned start; the learned policy is close to its teacher, not better.
- The safety filter that removes the last failures is an experiment outside the benchmark definition.
- Dense sensing is still sphere-based (optional rays occupy leftover obstacle slots); full-height mesh walls and corridors are out of scope. Default sensing is local and mostly ideal unless noisy evaluation is requested.
- One full training seed takes about 3-3.5 h on a laptop.

## Next steps

1. Run pipeline v4 in full (`caffeinate -i make training`) and read `models/showcase/SUMMARY.md`.
2. Train seeds 0 1 2 (`make train-full-multi`) and report all of them.
3. Decide how to report the safety filter (separate "learned + shield" controller, with tests).
4. Test generalization: other swarm sizes, noise/dropout, faster or larger objects.
5. Ablations: cloning only vs PPO only vs both; local vs central critic; with vs without crash replay.
6. Validate on full physics (gym-pybullet-drones) before any hardware work.
