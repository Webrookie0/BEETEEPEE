# Paper 03: QuadSwarm: A Modular Multi-Quadrotor Simulator for Deep Reinforcement Learning with Direct Thrust Control

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Zhehui Huang et al.. **QuadSwarm: A Modular Multi-Quadrotor Simulator for Deep Reinforcement Learning with Direct Thrust Control**. ICRA 2023 workshop.

[Paper / official source](https://arxiv.org/abs/2306.09537) · [Full text](https://arxiv.org/html/2306.09537)

**Relevant source sections:** §III QuadSwarm, particularly III-C observations, III-D scenarios and III-F RL interface; §IV simulation speed. The public repository is also a historical hyperparameter reference.

**Project status:** Architecture and experiment-design influence; simulator and training framework not imported.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Modular simulation and randomization

The project separates simulation (`env.py`), scenario generation (`scenarios.py`), controllers (`baseline.py`, `policy.py`), training (`train.py`), evaluation (`evaluate.py`) and rendering (`server.py`, `web/`). Seeded NumPy generators vary initial slot jitter, obstacle timing, angle, speed, radius and dense cluster placement. Counts, sensing and disturbances are configurable.

Conceptually $S_{t+1}=f(S_t,a_t;\xi)$ with scenario variables $\xi$ drawn by a seeded generator. This is a description of SwarmFlow's experiment setup, not a transplanted dynamics equation. The actual integrator is [documented here](project_context.md#flight-dynamics).

## Hyperparameter influence

[quad-swarm-rl](https://github.com/Zhehui-Huang/quad-swarm-rl/) was cited during pipeline v2 tuning: fine-tuning learning rate $10^{-4}$, no entropy bonus, and a wider policy. Current SwarmFlow widths, batching and budgets are listed in [configuration](configuration.md#training-defaults). They are engineering selections, not an exact reproduction of any repository revision; the older comparison was not pinned to a commit.

## Differences and scope

No QuadSwarm motor lag/noise, thrust allocation, rotor dynamics, downwash or contact response is implemented. The browser's rotor animation is illustrative. The code has no Sample Factory, distributed collector or external rendering dependency. Shared design principles do not transfer the paper's runtime, robustness or sim-to-real claims.

## Repository evidence

[Research plan](../docs/research-plan.md), [training §22](../training.md#22-hyperparameters-ours-compared-with-the-references), and [source module map](file_inventory.md#source-and-entry-points).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
