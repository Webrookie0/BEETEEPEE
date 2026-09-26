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

Stage 1 trains on `hold` and `follow`. Stage 2 resumes from the stage 1 checkpoint and adds `crossing` and `head_on`. The final checkpoint is `runs/stage2-seed0/latest.pt`.

The smoke checkpoint is only a machinery check. A serious result requires multiple independent training seeds, held-out evaluation seeds, comparison against the predictive baseline, and joint inspection of collision rate, clearance, formation error and recovery. Current development evidence is summarized in [docs/validation.md](docs/validation.md).

## How the paper conclusions were reached

The project began by reading six primary sources or official project pages and translating each source into either a design decision, a comparison baseline, a limitation or a future validation step. The conclusions were deliberately kept narrower than the papers' full claims:

1. Reuse ideas that are directly relevant to decentralized cooperative flight: shared policies, local relative observations, permutation-invariant neighbor processing and explicit formation measurements.
2. Separate stabilized swarm movement from low-level motor and attitude control so failures can be attributed to coordination and avoidance rather than an unmanageable combined optimization problem.
3. Treat moving spherical objects as the first benchmark scope. Dense static walls and corridors require richer distance or ray observations and should not be inferred from this task.
4. Use PPO as a credible learning baseline, but call the current method parameter-sharing independent PPO with a local critic. It is not MAPPO.
5. Use predictive relative-motion reasoning in the classical controller, while avoiding the claim that the controller is ORCA or has ORCA's formal guarantees.
6. Keep PyBullet and hardware validation as later stages. The current model is a reduced-order stabilized-flight benchmark, not a physical multirotor simulator.

These conclusions are hypotheses and engineering choices supported by the reviewed literature, not conclusions that the current small development runs prove. The full source notes are in [docs/research-plan.md](docs/research-plan.md).

## Paper-to-implementation map

### Batra et al. — Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning, CoRL 2021

The paper supports shared decentralized policies, local neighbor position/velocity information, permutation-aware set processing, PPO training and explicit formation re-establishment around moving obstacles. It also shows that the general behavior already has precedent, so SwarmFlow does not present the basic task as an entirely new idea.

Implemented in SwarmFlow:

- Shared neural policy weights across drones.
- Local self, neighbor and obstacle observations using relative position and velocity.
- Masked set encoders with mean and max pooling, making neighbor order irrelevant.
- PPO-based training with optional behavior-cloning warm start.
- Moving spherical obstacles and explicit recovery to each drone's original slot.
- Separate collision, formation-error and recovery measurements.

Not copied or claimed: the paper's direct-thrust setup, very large training budget, full reported performance and any claim that this implementation reproduces its results.

### Huang et al. — Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep Reinforcement Learning, ICRA 2024

This work motivated attention to obstacle representation, collision-state resets and the limitations of nearest-obstacle-only sensing. Its dense-obstacle setting is materially different from the current moving-object benchmark, so it led to a scope boundary: dense walls and corridors require a later sensing and navigation experiment.

Implemented or reflected:

- Bounded local obstacle observations with visibility masks and configurable sensor capacity.
- Optional observation noise and dropout for robustness experiments.
- Fresh on-policy PPO rollouts after cloning; stale transitions are not reused across resets.
- Swept collision detection and terminal failure metrics.
- A documented future path for fresh failure-state reset experiments.

Not yet implemented: the paper's attention architecture, its dense full-height static-obstacle benchmark and its complete replay/reset training method.

### Huang et al. — QuadSwarm: A Modular Multi-Quadrotor Simulator for Deep Reinforcement Learning with Direct Thrust Control, ICRA 2023 workshop

The paper and repository supported separating simulation, rendering, scenario generation and randomization. They are treated as references, not as the base code or as evidence of robustness.

Implemented in SwarmFlow:

- A standalone environment separated from the browser renderer.
- Reproducible seeded scenario generation.
- Configurable drone counts, scenarios, sensor limits and disturbances.
- A local web viewer that consumes simulation state through a Python server.

Not implemented: direct thrust dynamics or the QuadSwarm simulator itself.

### Yu et al. — The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games, NeurIPS 2022 Datasets and Benchmarks

This work supports treating cooperative PPO as a serious baseline, while not implying that game benchmarks establish drone performance. It informed the initial learning choice and the conservative naming of the algorithm.

Implemented in SwarmFlow:

- Parameter-sharing independent PPO for simultaneous drone actions.
- A local critic with the same information class as the actor.
- GAE, advantage normalization, ratio clipping, gradient clipping and approximate-KL early stopping.
- Checkpointing, resume support and seeded evaluation.

Not implemented: centralized MAPPO training, distributed-scale data collection or any claim of parity with the paper's environments.

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
- Eight scripted scenarios: hold, follow, crossing, head-on, repeated crossings, static spheres, turning target and wind gust.
- Local masked observations with optional noise and dropout.
- Predictive baseline, slot-only no-avoidance ablation and shared learned policy.
- Behavior cloning, clipped IPPO/PPO training, checkpoint save/reload/resume and seeded JSON evaluation.
- Browser viewer with live injections, trails, slot markers, telemetry and PNG export.
- Metrics for failure, collision, clearance, slot RMSE, centroid RMSE, smoothness and event recovery.

## Current limitations

The current results are development checks on a reduced-order model. The learned smoke checkpoint is not qualified, the baseline trials are small, most sensing is ideal, event geometry is scripted, and no physical-flight conclusion is justified. Publication-level comparisons, multi-seed training, dense-obstacle navigation, full physics and hardware validation remain future work.
