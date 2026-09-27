# SwarmFlow so far

This document records the current setup, execution flow, research conclusions and what has been implemented from each paper reviewed for SwarmFlow. It is a project status document, not a claim that the benchmark or learned policy is publication-ready.

## Run setup

SwarmFlow is a local Python project using Python 3.12, NumPy, Gymnasium, PettingZoo and PyTorch. The included `Makefile` is the recommended entry point.

```sh
cd /Users/sumit/Documents/Codex/developer/neww
make install
```

This creates or updates `.venv` using `uv`. To run the browser laboratory:

```sh
make run
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The viewer uses the predictive baseline unless a checkpoint is explicitly loaded.

Run the tests and baseline evaluation with:

```sh
make test
make eval
```

## Training flow

Run the short pipeline first to verify cloning, PPO updates, checkpoint saving/reloading and evaluation:

```sh
make train-smoke
make eval-smoke
```

Then run the staged experiment:

```sh
make train-stage1
make eval-stage1
make train-stage2
make eval-stage2
make view-learned
```

Stage 1 trains on `hold` and `follow`. Stage 2 resumes from the stage 1 checkpoint and adds `crossing` and `head_on`. The seed-0 checkpoint is `runs/stage2-seed0/latest.pt`. Repeat independent seeds with `make train-stage1-multi` / `eval-stage1-multi` / `train-stage2-multi` / `eval-stage2-multi` (`SEEDS` default `0 1 2`). Additional checks: `make eval-heldout`, `make eval-noisy`, `make eval-dense`.

The smoke checkpoint is only a machinery check. A serious result requires multiple independent training seeds, held-out evaluation seeds, comparison against the predictive baseline, and joint inspection of collision rate, clearance, formation error and recovery. Current development evidence is summarized in [docs/validation.md](docs/validation.md).

## How the paper conclusions were reached

The project began by reading six primary sources or official project pages and translating each source into either a design decision, a comparison baseline, a limitation or a future validation step. The conclusions were deliberately kept narrower than the papers' full claims:

1. Reuse ideas that are directly relevant to decentralized cooperative flight: shared policies, local relative observations, permutation-invariant neighbor processing and explicit formation measurements.
2. Separate stabilized swarm movement from low-level motor and attitude control so failures can be attributed to coordination and avoidance rather than an unmanageable combined optimization problem.
3. Treat moving spherical objects as the first benchmark scope. Dense static walls and corridors require richer distance or ray observations and should not be inferred from this task.
4. Use PPO as a credible learning baseline. Default training is parameter-sharing IPPO with a local critic; `--critic central` is an optional MAPPO-style comparison, not a claim of full MAPPO.
5. Use predictive relative-motion reasoning in the classical controller, while avoiding the claim that the controller is ORCA or has ORCA's formal guarantees.
6. Keep PyBullet and hardware validation as later stages. The current model is a reduced-order stabilized-flight benchmark, not a physical multirotor simulator.

These conclusions are hypotheses and engineering choices supported by the reviewed literature, not conclusions that the current small development runs prove. The full source notes are in [docs/research-plan.md](docs/research-plan.md).

## Paper-to-implementation map

### Batra et al. — Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning, CoRL 2021

The paper supports shared decentralized policies, local neighbor position/velocity information, permutation-aware set processing, PPO training and explicit formation re-establishment around moving obstacles. It also shows that the general behavior already has precedent, so SwarmFlow does not present the basic task as an entirely new idea.

Implemented in SwarmFlow:

- Shared neural policy weights across drones.
- Local self, neighbor and obstacle observations using relative position and velocity.
- Masked attention set encoders, making neighbor order irrelevant.
- PPO-based training with optional behavior-cloning warm start.
- Moving spherical obstacles and explicit recovery to each drone's original slot.
- Separate collision, formation-error and recovery measurements.

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
- Optional `--failure-reset` (pre-collision state restore, fresh on-policy rollouts) and `--critic central` (MAPPO-style shared value).

Not yet implemented: the paper's dense full-height static-obstacle benchmark and its full training stack.

### Huang et al. — QuadSwarm: A Modular Multi-Quadrotor Simulator for Deep Reinforcement Learning with Direct Thrust Control, ICRA 2023 workshop

The paper and repository supported separating simulation, rendering, scenario generation and randomization. They are treated as references, not as the base code or as evidence of robustness.

Implemented in SwarmFlow:

- A standalone environment separated from the browser renderer.
- Reproducible seeded scenario generation, including randomized event times and headings.
- Configurable drone counts, scenarios, sensor limits and disturbances.
- A local web viewer that consumes simulation state through a Python server.

Not implemented: direct thrust dynamics or the QuadSwarm simulator itself.

### Yu et al. — The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games, NeurIPS 2022 Datasets and Benchmarks

This work supports treating cooperative PPO as a serious baseline, while not implying that game benchmarks establish drone performance. It informed the initial learning choice and the conservative naming of the algorithm.

Implemented in SwarmFlow:

- Parameter-sharing independent PPO for simultaneous drone actions (default local critic).
- Optional centralized critic over mean-pooled agent embeddings (`--critic central`); not a claim of MAPPO parity.
- GAE, advantage normalization, ratio clipping, gradient clipping and approximate-KL early stopping.
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

- A sampled receding-horizon predictive baseline.
- Relative position and velocity in local observations.
- Predicted finite-response motion over a short horizon.
- Clearance penalties, detours and waiting behavior in the baseline.
- Continuous swept collision checks over integration segments.

Not implemented: ORCA's reciprocal velocity-obstacle formulation, linear program or formal guarantees.

## What is implemented overall

- PettingZoo parallel environment with Gymnasium spaces.
- Immutable target-relative formation slots.
- Stabilized 3D translational dynamics with finite response and acceleration limits.
- Swept drone/drone and drone/object collision checks.
- Eight scripted scenarios plus `dense`: hold, follow, crossing, head-on, repeated crossings, static spheres, a dense sphere cluster, turning target and wind gust.
- Local masked observations with optional noise, dropout, and optional ray hits in leftover obstacle slots.
- Predictive baseline, slot-only no-avoidance ablation and shared learned policy.
- Behavior cloning, clipped IPPO/PPO training, checkpoint save/reload/resume and seeded JSON evaluation.
- Browser viewer with live injections, trails, slot markers, telemetry and PNG export.
- Metrics for failure, collision, clearance, slot RMSE, centroid RMSE, smoothness and event recovery, plus aggregate success rate and slot-RMSE CI; `swarmflow compare` diffs two reports.
- Makefile multi-seed stage targets, held-out eval (`seeds 100000+`), noisy eval, and dense eval.

## Current limitations

Hardware flight, full 6DOF/PyBullet rotor physics, and ORCA's linear program are not implemented. Publication-scale training budgets remain unfinished until multi-seed stage runs are actually executed. Dense sensing is still sphere-based (optional rays occupy leftover obstacle slots); full-height mesh walls and corridors are out of scope. Default sensing is still local and mostly ideal unless noisy eval is requested. The smoke checkpoint is not a qualified policy.
