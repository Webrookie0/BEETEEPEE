# SwarmFlow research and implementation context

Read this file from **Section 1 onward** to follow the implementation from problem definition and environment construction through training, evaluation and viewing. Every section links directly to the relevant heading inside a numbered paper note or the project-equation notes.

The folder documents **30 research papers**, plus cited software/docs/tutorials, all implemented equations and settings, optional/rejected ideas, historical experiments, and an index of every retained project file. It describes the code checked on **29 September 2026**: pipeline **v4**, checkpoint format **swarmflow-v4**. Existing older notes are preserved and reconciled in [history.md](history.md#discrepancies-resolved-by-current-code).

The numbered paper files are `paper_01.md` through `paper_30.md`; the first six retain the order of the original research plan. A paper's **status** distinguishes an implemented method from an adaptation, background reference, disabled option or future work. Equations are expressed in consistent project notation and labelled when they describe a local adaptation rather than the original paper.

## Folder guide

- [Project context](project_context.md): task, units, formation, scenarios, dynamics, sensing, planner, reward, recovery, metrics, training and viewer.
- [Configuration](configuration.md): every environment/training default, curriculum overrides, commands and dependencies.
- [History](history.md): archived results, version changes, contradictions and unavailable historical artifacts.
- [File inventory](file_inventory.md): every retained source, test, document, model, log, report, image and trajectory file, with clickable paths.
- [Other references](references.md): PPO implementation blog, textbook/tutorials, Gymnasium, PettingZoo, QuadSwarm repository and verification notes.
- [Paper index](#paper-index): one source-linked note per research paper.

Large raw data and model binaries remain at their original paths and are linked from this folder. Generated dependency/cache and Git internals are explained, not copied. This is comprehensive context for the available checkout, not a claim to possess legacy source files that are absent.

## Implementation sequence

The sequence below is a dependency and execution walkthrough, not a claim about historical commit order. Within each training stage, the precise loop order is in [training execution](project_context.md#training-execution-and-storage).

### Section 1: Define the task and research scope

The implementation begins with a following/yielding/recovery task and velocity-level abstraction.

- [Paper 1: formation recovery precedent](paper_01.md#persistent-formation-and-moving-obstacles)
- [Paper 5: controller-level interface](paper_05.md#stabilized-velocity-interface)
- [Project task contract](project_context.md#task-and-boundaries)

### Section 2: Configure the simulator and seeded scenarios

Create Config, choose a scenario/world/geometry, initialize persistent offsets and reset state.

- [Paper 3: modular simulation and randomization](paper_03.md#modular-simulation-and-randomization)
- [Formation equations](project_context.md#formation-geometry)
- [Target/object motion and scenario constants](project_context.md#scenarios-and-worlds)
- [Terrain and continuous-viewer extensions](project_context.md#terrain-and-continuous-viewing)
- [All environment defaults](configuration.md#environment-defaults)

### Section 3: Advance flight and detect collisions

Commands pass through finite-response dynamics; swept sphere checks and ground/world tests identify failure.

- [Integrator and limits](project_context.md#flight-dynamics)
- [Paper 6 context: relative-motion geometry](paper_06.md#swept-collision-geometry)
- [Collision attribution and boundary behavior](project_context.md#collision-and-world-failure)

### Section 4: Construct local observations

Build self state and bounded masked neighbor/object sets with optional position noise, dropout and ray hits.

- [Paper 1: shared local observations](paper_01.md#shared-local-policy-and-relative-observations)
- [Paper 2: obstacle sensing adaptation](paper_02.md#bounded-obstacle-sensing)
- [Optional gap/proximity/closing features](paper_02.md#optional-gap-features)
- [Exact tensor order](project_context.md#observations)
- [Ray intersection equations](project_context.md#ray-intersections)

### Section 5: Build the predictive teacher and comparison controllers

Sample candidate velocities, predict tracking/clearance costs and choose a command. The teacher uses larger penalty margins.

- [Paper 6: predictive relative motion](paper_06.md#relative-motion-reasoning)
- [Paper 28: teacher margin rationale](paper_28.md#clearance-margin-for-imitation-error)
- [Paper 29: optional cost-weighted blend, off](paper_29.md#temperature-weighted-commands)
- [Complete planner cost and slot-only ablation](project_context.md#predictive-controller)

### Section 6: Compute rewards and record recovery

Accumulate per-drone shaped rewards and event/episode statistics after each action.

- [Paper 1: reward structure and per-drone penalties](paper_01.md#reward-components-and-collision-attribution)
- [Alive-reward and API documentation](references.md#gymnasium-and-pettingzoo)
- [Complete reward formula and bounds](project_context.md#reward)
- [Settling and overlapping-event timing](project_context.md#recovery)

### Section 7: Encode the observation as sets

Embed each row, pool with a learned query and combine the pools with self features.

- [Paper 18: Deep Sets principle](paper_18.md#order-independent-neighbor-sets)
- [Paper 19: attention equation](paper_19.md#attention-pooling)
- [Exact network dimensions and normalization](paper_19.md#complete-encoder)

### Section 8: Construct the actor and critic

Use one shared stochastic actor, a separate local value encoder by default, and optional team-pooled critic.

- [Paper 9: default IPPO](paper_09.md#independent-ppo)
- [Paper 4: optional central value](paper_04.md#optional-centralized-value-function)
- [Paper 20: CTDE concept](paper_20.md#centralized-training-decentralized-execution)
- [Paper 10: network separation/initialization](paper_10.md#separate-networks-and-initialization)
- [Action distribution and corrected log likelihood](paper_07.md#squashed-gaussian-actions)

### Section 9: Collect noisy teacher demonstrations

At the start of each stage, execute noisy teacher actions but save clean teacher targets.

- [Paper 14: DART-inspired collection](paper_14.md#noisy-teacher-clean-labels)
- [Cloning objective and optimizer](paper_13.md#cloning-loss-and-fitting)

### Section 10: Aggregate student-visited demonstrations

Run three further student-driven rounds and refit on all stage data collected so far.

- [Paper 13: DAgger recurrence](paper_13.md#dataset-aggregation)
- [Optional weighted sampling, uniform by default](paper_13.md#optional-hard-state-weighting)

### Section 11: Validate the cloned policy and prepare the critic

Save a starting validation candidate, initialize rollout workers and train the value function before actor updates.

- [Paper 16 context: critic warm-up](paper_16.md#critic-warm-up-motivation)
- [Paper 17: fine-tuning limitations](paper_17.md#fine-tuning-context)
- [Paper 30: preserve the cloned candidate](paper_30.md#retaining-pretrained-behavior)
- [Exact phase order and counters](project_context.md#training-execution-and-storage)

### Section 12: Collect on-policy experience and replay collision starts

Batch inference across environments, keep raw actions/old log likelihoods and reset or restore workers at episode boundaries.

- [PPO implementation reference](references.md#ppo-implementation-details)
- [Paper 2: fresh rollout from saved states](paper_02.md#pre-collision-state-replay)
- [Paper 12: terminal versus time-limit masks](paper_12.md#termination-and-truncation)

### Section 13: Scale rewards and estimate advantages

Compute discounted-return scale, final-observation bootstrap, GAE and batch-normalized advantages.

- [Paper 11: exact running moments](paper_11.md#reward-scaling)
- [Distinction from paper value normalization](paper_04.md#normalization-distinction)
- [Paper 8: GAE and return targets](paper_08.md#gae-recurrence)

### Section 14: Optimize PPO while retaining teacher behavior

Apply clipped policy loss, value loss, demonstration MSE, clipping, schedules and KL stopping.

- [Paper 7: PPO objective](paper_07.md#clipped-surrogate)
- [Exact combined loss coefficients](paper_07.md#combined-loss)
- [Paper 15: DAPG-inspired demonstration term](paper_15.md#demonstrations-before-and-during-rl)
- [Paper 27: Kickstarting relation](paper_27.md#teacher-loss-during-reinforcement-learning)
- [Adam, gradient clipping and schedules](paper_11.md#optimizer-and-schedules)
- [KL estimator and stop timing](paper_07.md#approximate-kl-and-stopping)
- [Exploration floor and stage initialization](paper_10.md#exploration-settings)

### Section 15: Save checkpoints and move through the curriculum

Preserve the best selection result, finish stage budgets and start the next stage from best.pt.

- [Paper 22: easy-to-hard task sets](paper_22.md#progressive-task-distribution)
- [Paper 23: transfer and lineage](paper_23.md#tasks-sequencing-and-transfer)
- [Paper 21: scale context and step units](paper_21.md#budget-context)
- [Current v4 budgets and overrides](configuration.md#curriculum-stages)
- [Checkpoint contents and resume semantics](project_context.md#training-execution-and-storage)

### Section 16: Select and evaluate the showcase

Choose using selection seeds, compare learned/baseline on matching test seeds and apply explicit acceptance checks.

- [Paper 24: seed separation](paper_24.md#seed-separation)
- [Model selection and test use](paper_24.md#selection-and-testing)
- [Paper 25: multiple training seeds](paper_25.md#multiple-seeds-and-reproducibility)
- [Paper 26: uncertainty limits](paper_26.md#uncertainty-and-complete-reporting)
- [Failure/recovery denominators](paper_26.md#failure-and-recovery-denominators)
- [Every metric, ranking score and acceptance threshold](project_context.md#metrics-and-selection)

### Section 17: Present the runtime and inspect experiments

The browser displays actual controller behavior, training history and stored test evidence.

- [Viewer/API/rendering context](project_context.md#viewer-and-application-flow)
- [Commands and launcher](configuration.md#commands-and-entry-points)
- [Saved measured results](history.md#saved-training-runs)
- [Version history](history.md#pipeline-changes)
- [Unavailable experiment artifacts](history.md#missing-and-historical-only-context)

### Section 18: Trace all folders and remaining research

Use the inventory for every retained artifact and distinguish implemented methods from pending research studies.

- [Every source module and function](file_inventory.md#source-and-entry-points)
- [Every test function](file_inventory.md#tests)
- [All model/history files](file_inventory.md#models)
- [All development runs](file_inventory.md#runs)
- [Every legacy result artifact](file_inventory.md#results)
- [Legacy graph labels](file_inventory.md#graphify-metadata)
- [Libraries and build context](configuration.md#dependencies-and-build)
- [Tests and remaining experiments](project_context.md#tests-and-unimplemented-experiments)
- [Source-verification coverage](references.md#verification-and-scope)

## Paper index

Each note includes the original paper link, relevant source sections, every identified contribution, equations or links to shared equations, code locations and attribution limits.

1. [Paper 01: Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning](paper_01.md) — Implemented with substantial task, dynamics, architecture and reward adaptations.
2. [Paper 02: Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep Reinforcement Learning](paper_02.md) — Adapted obstacle representation and collision-state replay; optional rays/gap features; dense full-height benchmark not reproduced.
3. [Paper 03: QuadSwarm: A Modular Multi-Quadrotor Simulator for Deep Reinforcement Learning with Direct Thrust Control](paper_03.md) — Architecture and experiment-design influence; simulator and training framework not imported.
4. [Paper 04: The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games](paper_04.md) — Cooperative PPO design reference; default local critic, optional mean-pooled central critic; paper value normalization not implemented.
5. [Paper 05: Learning to Fly — a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control](paper_05.md) — Future physics-validation reference, not an installed simulation backend.
6. [Paper 06: Reciprocal n-body Collision Avoidance](paper_06.md) — Relative-motion motivation for a sampled predictive heuristic; ORCA itself is not implemented.
7. [Paper 07: Proximal Policy Optimization Algorithms](paper_07.md) — Implemented clipped PPO with project-specific scaling, stopping, imitation loss and continuous action distribution.
8. [Paper 08: High-Dimensional Continuous Control Using Generalized Advantage Estimation](paper_08.md) — Implemented GAE with separate failure-bootstrap and reset-continuation masks.
9. [Paper 09: Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?](paper_09.md) — Default algorithm-family choice: parameter-sharing independent PPO with a local critic.
10. [Paper 10: What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study](paper_10.md) — Implementation recommendations adapted to this controller.
11. [Paper 11: Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](paper_11.md) — Motivation for documented reward scaling, scheduling, clipping and initialization; exact choices are project-specific.
12. [Paper 12: Time Limits in Reinforcement Learning](paper_12.md) — Implemented distinction between failure termination and artificial time-limit truncation.
13. [Paper 13: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](paper_13.md) — Implemented student-driven dataset aggregation after a noisy-teacher initial round.
14. [Paper 14: DART: Noise Injection for Robust Imitation Learning](paper_14.md) — Fixed-noise teacher data collection inspired by DART; the noise optimization procedure is not implemented.
15. [Paper 15: Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations](paper_15.md) — Imitation warm start and demonstration regularization adapted to PPO, not the full DAPG optimizer.
16. [Paper 16: Jump-Start Reinforcement Learning](paper_16.md) — Fine-tuning motivation; JSRL guide-policy roll-in algorithm is not implemented.
17. [Paper 17: AWAC: Accelerating Online Reinforcement Learning with Offline Datasets](paper_17.md) — Background for fine-tuning from demonstrations; AWAC algorithm not implemented.
18. [Paper 18: Deep Sets](paper_18.md) — Permutation-invariant set-processing principle implemented through attention pooling.
19. [Paper 19: Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks](paper_19.md) — Learned-query multihead attention pooling, simplified from the full Set Transformer.
20. [Paper 20: Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](paper_20.md) — Centralized-training/decentralized-execution concept only; MADDPG is not implemented.
21. [Paper 21: Champion-level drone racing using deep reinforcement learning](paper_21.md) — Background and training-scale comparison only.
22. [Paper 22: Curriculum Learning](paper_22.md) — Easy-to-hard scheduling principle; fixed, hand-authored task stages.
23. [Paper 23: Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey](paper_23.md) — Framework for describing task generation, sequencing and transfer; no automatic curriculum algorithm implemented.
24. [Paper 24: Leveraging Procedural Generation to Benchmark Reinforcement Learning](paper_24.md) — Separate training, selection and test generation; Procgen environments not used.
25. [Paper 25: Deep Reinforcement Learning that Matters](paper_25.md) — Reproducibility, multiple seeds and explicit settings; not a performance guarantee.
26. [Paper 26: Deep Reinforcement Learning at the Edge of the Statistical Precipice](paper_26.md) — Careful evaluation motivation; bootstrap/IQM/performance-profile toolbox not implemented.
27. [Paper 27: Kickstarting Deep Reinforcement Learning](paper_27.md) — Teacher regularization with a changing weight; adapted loss and schedule.
28. [Paper 28: Robust Model Predictive Control of Constrained Linear Systems with Bounded Disturbances](paper_28.md) — Qualitative motivation for a larger teacher clearance margin; robust/tube MPC not implemented.
29. [Paper 29: Information Theoretic MPC for Model-Based Reinforcement Learning](paper_29.md) — Optional MPPI-style cost-weighted candidate blend; tested and disabled in the current recipe.
30. [Paper 30: Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem](paper_30.md) — Diagnosis and retention motivation; no separate paper-specific fine-tuning algorithm.


## How to cite and interpret this project

Use each note's **Source** section for the publication and its method sections for what this project actually adopts. Cite source papers for their ideas and SwarmFlow source/configuration for local coefficients, benchmark thresholds and modified formulas. For example, PPO clipping is from Paper 7; the 0.7/0.3 m teacher margins and 5 s recovery threshold are project choices.

Pure learned inference uses the network alone. The predictive teacher is not ORCA; the PPO demonstration term is not full DAPG/Kickstarting; the optional central critic is not full MADDPG/MAPPO; and the stabilized model is not physical drone validation. Those boundaries are spelled out per paper so literature claims are not accidentally presented as project results.

For maintenance and source-check limitations, see [verification](references.md#verification-and-scope) and [updating the notes](references.md#updating-these-notes).
