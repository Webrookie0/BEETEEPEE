# How SwarmFlow trains its drone policy

> **Current version: 3.** After the first full training run, stages 2 and 3 were fixed (**Part C**). After the second run, the teacher was made more cautious and slalom was removed (**Part D**, at the end). Where a later part differs from an earlier one, the later part is current. Compared with the first sections, the main differences are:
> - cloning in every stage, with a teacher that keeps 0.25–0.6 m clearance;
> - an imitation loss inside PPO whose weight goes from 1.0 to 0.5;
> - an exploration floor, a per-drone crash penalty and a 1 m obstacle warning;
> - a wider network;
> - 7 scenarios instead of 8 (no slalom);
> - a total time of about 6 h.

This document explains what the training does, why the old training finished in 2–5 minutes without learning, what was changed (and which research each change comes from), and how every part of the code works. It is written for preparing the presentation. The practical "which command do I run" steps are in [guide.md](guide.md).

---

## 1. Short answers

**Was a pre-trained model used?** No pre-trained weights are downloaded or imported anywhere. The network is always created with random weights (`ActorCritic(...)` in `src/swarmflow/policy.py`) and trained on our machine. Three things made it *look* pre-trained:

1. The viewer and `Launch SwarmFlow.command` used to show only the **hand-coded predictive controller** (`PredictiveController` in `baseline.py`). The good-looking demo was never a neural network.
2. Some small checkpoints were already lying in `runs/` (for example `runs/smoke-policy/latest.pt`). They came from 2–5 minute runs and do not fly correctly.
3. Training starts with **behavior cloning**: the network first imitates our own hand-coded controller, then improves with reinforcement learning (PPO). That is still training by us, from our own code. If you want pure RL from random weights, see section 6.4.

**Why did training finish in 2–5 minutes?** The budget was tiny and the training itself was broken, so a longer run would not have fixed it. Section 2 shows the evidence from the saved logs, and section 3 lists the causes.

**What does the new training do?** It is a three-stage curriculum. It starts with behavior cloning plus DAgger, then runs PPO with 8 parallel environments and a separate critic network. The model is validated every 100,000 steps and the best checkpoint is saved as `best.pt`. At the end, the best run is tested on seeds it has never seen and stored in `models/showcase/`, where the viewer loads it. On this Mac the full pipeline takes about 3 hours for one seed.

### How to run the training

```sh
make test                          # once: all unit tests must pass
caffeinate -i make training        # full training in the terminal (python -m swarmflow pipeline)
caffeinate -i make run train       # the same training, plus the viewer with live progress (python -m swarmflow pipeline --view)
make run                           # afterwards: viewer with the best model (python -m swarmflow view)
```

- **Order**: both training commands run stage 1 → 2 → 3, then the showcase test.
- **Results**: each stage is stored in `models/stageN-seedS/`, and the chosen best model goes to `models/showcase/best.pt`, which the viewer uses.
- **Interruptions**: if training stops, run the same command again and it continues.

[guide.md](guide.md) has the details, the folder layout and a **Do and don't** list (section 6). The most important don'ts:

- don't delete or edit files in a stage folder while it trains;
- don't run two trainings into the same folder;
- don't change `--drones` or the other environment settings between stages;
- don't tune on the held-out test seeds;
- don't shorten the budgets.

---

## 2. What the old training actually did (evidence from the saved runs)

| Run | Budget | Wall time | Result |
| --- | --- | --- | --- |
| `runs/tests/09-stage1-seed0` (hold, follow) | 2,000 cloning steps + 50,000 PPO steps | 172 s including evaluation | Slot error **7.86 m** on hold and 1.38 m on follow; the baseline scores 0.014 m on the same seeds. |
| `runs/stage1-seed0` → `runs/stage2-seed0` | 50,000 + 100,000 PPO steps | about 2.5 min | **160 of 160** evaluation episodes failed with `world_boundary`: the drones flew out of the world after about 13.6 s. |

Inside the stage 1 training log (`runs/tests/09-stage1-seed0/training.jsonl`):

- **212 of 252 training episodes ended in a collision (84%)**, even though `hold` and `follow` contain no obstacles. The drones crashed into each other.
- `value_loss` stayed between **700 and 3,000**.
- `approx_kl` was about **0.0005** per update, which means the policy was barely changing.

A policy that is not changing, while its collisions stay high and its critic cannot fit, is not learning. Running it longer would only have produced a longer failure.

---

## 3. Root causes and fixes

| # | Problem in the old code | Effect | Fix | Where |
| --- | --- | --- | --- | --- |
| 1 | Only 50k + 100k environment steps in total (1.2 M drone transitions), from **one** environment, with 1,024 samples per update. | A few minutes of training on very noisy updates. | 5 M environment steps (40 M drone transitions) over 3 stages, from **8 parallel environments** giving 16,384 samples per update. | `curriculum.py` `STAGES`, `COMMON`; `train.py` `Worker` |
| 2 | Actor and critic **shared one network**, and the value loss (up to 3,000) was clipped together with the policy gradient to a combined norm of 0.5. | The value gradient took almost the whole clipping budget, so policy updates were tiny (KL ≈ 0.0005). The critic also overwrote the features the behavior-cloned actor relied on. | **Separate critic network**, with gradients clipped separately for actor and critic. | `policy.py` `ActorCritic(separate_critic=True)`; `train.py` update loop |
| 3 | Reward was **always negative and unbounded** (quadratic slot error), with a −200 crash penalty. | Once the drones drifted about 6 m, crashing cost less than continuing, so the policy learned to end episodes. This is why stage 2 failed 160/160 with `world_boundary`. | Every cost is **capped**, and each drone gets **+1 per second alive**. Surviving now always beats failing (proof in 5.2). | `env.py` `_reward` |
| 4 | Behavior cloning saw only the teacher's near-perfect states (slot error about 0.014 m, final MSE 1.8×10⁻⁶). | The network had never seen a drone away from its slot. At the first deviation it acted randomly and the errors compounded (the DAgger paper's covariate shift). | **DART noise injection plus DAgger rounds**: the student flies and the teacher labels the states it visits. | `train.py` `clone` |
| 5 | Exploration noise std 0.37, which is ±1.1 m/s every 50 ms, against a surface gap of only 0.89 m between neighbouring drones. | 84% of training episodes collided in empty scenarios. | Initial std **0.2** after cloning (Andrychowicz et al. recommend a lower initial std), which PPO can then adapt. | `train.py` `init_log_std` |
| 6 | PPO started with a **random critic** right after cloning. | Early advantages were noise, so the first updates damaged the cloned policy (reported in the Jump-Start RL and AWAC papers). | **Critic warm-up**: the first 10 updates after cloning train only the critic, with the actor frozen. Stages 2 and 3 start with 5 critic-only updates, because their new obstacle states are unknown to the critic. | `train.py` `critic_warmup` |
| 7 | No reward normalization, while returns ranged over hundreds of units. | Value targets were huge and unstable. | **Reward scaling** by a running standard deviation of the discounted return. | `train.py` `RewardScaler` |
| 8 | Only `latest.pt` was saved, overwritten after every update, and there was no model selection. | The final weights could be worse than earlier ones, and there was no "best" model to show. | Validation every 100k steps on dedicated seeds; the **best checkpoint is kept as `best.pt`**. Stage 2 starts from stage 1's `best.pt`, and the viewer shows the showcase `best.pt`. | `train.py` `validate`; `curriculum.py` |
| 9 | Constant learning rate. | Updates stay large and noisy at the end. | Linear annealing from 3×10⁻⁴ down to 3×10⁻⁵. | `train.py` |
| 10 | The same seeds were used for tuning and reporting. | Results were optimistic. | Separate seed ranges: training 0–999, checkpoint selection 5000+, development 1000+, **held-out test 100000+**. The test seeds are used once, after the model has been chosen. | `train.py`, `curriculum.build_showcase` |

---

## 4. The new training pipeline at a glance

```mermaid
flowchart TD
    A[Random network weights] --> B[Behavior cloning round 0<br/>teacher flies with action noise]
    B --> C[DAgger rounds 1-3<br/>student flies, teacher labels]
    C --> D[Critic warm-up<br/>10 updates, actor frozen]
    D --> E[Stage 1 PPO, 1M steps<br/>hold / follow / turn]
    E -->|best.pt| F[Stage 2 PPO, 2M steps, 5 critic-only updates first<br/>+ crossing / head-on objects]
    F -->|best.pt| G[Stage 3 PPO, 2M steps, 5 critic-only updates first<br/>+ repeated crossings / spheres / wind]
    G -->|best.pt of the best seed| H[Showcase: held-out test vs baseline<br/>models/showcase/]
    H --> I[Viewer loads models/showcase/best.pt<br/>and shows the verdict]
    V -. newest best.pt .-> I
    E -. every 100k steps .-> V[Validation on seeds 5000+<br/>save best.pt if improved]
    F -.-> V
    G -.-> V
```

| Stage | Scenarios | PPO steps | Starts from | Purpose |
| --- | --- | --- | --- | --- |
| 1 | hold, follow, turn | 1,000,000 | random weights + cloning (4 rounds × 10,000 steps) | Keep the formation and track a moving, turning target. |
| 2 | stage 1 + crossing, head_on | 2,000,000 | `models/stage1-seedS/best.pt` | Make room for moving objects and return to the same slots. |
| 3 | stage 2 + double_crossing, slalom, gust | 2,000,000 | `models/stage2-seedS/best.pt` | The full eight-scenario benchmark. |

Shared settings (`COMMON` in `curriculum.py`): 8 environments, rollout horizon 256 steps per environment, 5 PPO epochs, minibatch 2,048, learning rate 3×10⁻⁴ annealed to 10%, validation every 100,000 steps on 5 seeds per scenario, and a snapshot every 500,000 steps.

**Measured time on this Mac** (M-series, 4 PyTorch threads). Rollouts run at about 900–1,200 environment steps per second, and one PPO update takes about 1.5 s, so training runs at roughly 580 steps/s overall. Validation runs at about 0.6 s per episode.

| Part | Approximate time |
| --- | --- |
| Behavior cloning + DAgger | 5–7 min |
| Stage 1 | 30–40 min |
| Stage 2 | about 1 h |
| Stage 3 | about 1 h |
| Showcase test (8 scenarios × 20 held-out seeds, learned + baseline) | 7–10 min |
| **Total for one seed** | **about 3 h** |

The log prints the real `steps/s` and an ETA after every update.

Why this budget: the quadrotor-swarm work this project builds on (Batra et al., 2021) trained each policy on about a billion samples, and Swift, the champion-level racing drone (Kaufmann et al., 2023), used on the order of 10⁸ simulated interactions. Our task is much simpler: velocity commands above a stabilized flight model, not raw thrust. So 5 M team steps (40 M drone transitions), plus a behavior-cloning warm start, is a realistic laptop budget. It is still a starting budget, not a guarantee of convergence. Validation (section 5.8) tells you whether it was enough.

---

## 5. How each part works (code walkthrough)

### 5.1 Environment: `src/swarmflow/env.py`, `scenarios.py`, `config.py`

- `SwarmEnv` is a PettingZoo `ParallelEnv`. Every drone acts at the same time at 20 Hz (`dt = 0.05 s`), and an episode lasts 24 s (480 steps).
- **Action**: 3 numbers in [−1, 1], a desired velocity scaled by 3 m/s. The flight model follows it with a 0.28 s response and a 5 m/s² acceleration limit (`step`).
- **Observation** (`_observe`), per drone:
  - `self` (17 values): vector to its own slot, own velocity, target velocity, previous command, slot offset, altitude and remaining time.
  - `neighbors` (6×8): the nearest drones within 7 m, as relative position, relative velocity, radius and a valid flag.
  - `obstacles` (6×8): the same schema for external objects.
  - Everything is local and relative, with no global state.
- **Collision checking** is swept over each 10 ms sub-step (`_clearance`), so fast objects cannot tunnel through.
- **Recovery** (`_update_recovery`) counts a disturbance as recovered once every drone is within 0.35 m of its **own** slot, at below 0.35 m/s velocity error, for 1 s after the object leaves.

### 5.2 Reward: `SwarmEnv._reward`

For each drone and step (`dt = 0.05`), with slot error `e` capped at 5 m:

```text
tracking = e²           if e ≤ 1 m
           2e − 1       if e > 1 m        (linear far away: a constant pull back, never exploding)
cost     = tracking + 0.12·min(v_err, 3)² + 6·proximity + 0.005·|u|² + 0.001·min(|Δu|/dt, 20)²
reward   = dt · (1 − cost)                (+1 per second alive, minus the costs)
failure  : reward −= 100                  (collision, ground, or leaving the world ends the team's episode)
```

- **proximity** penalises being closer than 0.45 m (surface to surface) to the nearest drone or to any object.
- **u** is the velocity command and **Δu** its change between steps, which rewards smooth commands.
- **Why surviving always wins.** The worst possible "alive" step costs about 0.6. With discount γ = 0.99, even an endless run of worst steps sums to about −60, which is still better than the −100 failure. So ending an episode is never the cheapest option, which fixes the old `world_boundary` behaviour. The MuJoCo locomotion benchmarks use a per-step "healthy" bonus for the same reason. The unit test `test_reward_is_bounded_and_failure_is_always_worse` checks this.
- **Near the slot** (e ≤ 1 m) the tracking term is the same `e²` as before, so formation precision is still rewarded exactly as in the old reward.

### 5.3 Network: `src/swarmflow/policy.py`

One network is shared by all drones (parameter sharing). Every drone runs the same weights on its own local observation.

- `SetEncoder` embeds each neighbour or obstacle row (7→32→64, tanh), then pools the set with **multi-head attention from a learned query**. This is "pooling by multihead attention" from the Set Transformer. Padded rows are masked, so the result does not depend on the order or number of neighbours. Batra et al. (2021) use the same idea (deep sets or attention) for quadrotor swarms.
- `Encoder` concatenates `own` (17→64), the neighbour set (64) and the obstacle set (64), then a trunk 192→128→64.
- `ActorCritic(Encoder)` holds the parts that act and judge:
  - **Actor**: `actor` layer 64→3 gives the mean. `log_std` is a learned per-dimension standard deviation. Actions are `tanh(Normal(mean, std))`, a squashed Gaussian that keeps commands inside [−1, 1]. `log_probability` applies the tanh change-of-variables correction.
  - **Critic**: with `separate_critic=True` (the new default) the critic has **its own `Encoder`** (`critic_encoder`) and head 64→1, so value errors cannot distort the actor. Andrychowicz et al. (2021) found separate policy and value networks work better; the "37 implementation details" guide uses them for continuous control.
  - Sizes: actor 72,454 parameters, critic 72,321.
  - The final actor layer starts with tiny weights (orthogonal init, gain 0.01), so the first actions are close to zero, as recommended by Andrychowicz et al.
- `act()` returns the deterministic action `tanh(mean)`, used for evaluation and in the viewer.
- `ModelController` flies a team with a model already in memory. `LearnedController` loads a checkpoint file.
- `--critic central` gives a MAPPO-style centralized value (the mean over the team's features); the default is a local critic (IPPO). de Witt et al. (2020) and Yu et al. (2022) showed that independent PPO with parameter sharing is a strong cooperative baseline.

### 5.4 Behavior cloning with DAgger: `train.clone`, `train._fit_clone`

1. **Round 0 (DART)**: the hand-coded `PredictiveController` flies, with Gaussian noise (std 0.3) added to the commands it *executes*. Its *clean* commands are stored as labels. The data therefore includes states away from the slots together with the correct way back (Laskey et al., 2017).
2. **Rounds 1–3 (DAgger)**: the *student network* flies. The teacher labels every state the student visits, and the network is refit on all data collected so far (Ross et al., 2011). This targets exactly the mistakes the student makes.
3. Loss: mean squared error between `tanh(mean)` and the teacher's command, trained with Adam (lr 5×10⁻⁴, minibatch 1,024, 15 epochs, gradient clipping 0.5). Only actor parameters are trained. Stage 1 collects 4 × 10,000 steps × 8 drones = **320,000 labelled samples**.

Each round writes a `behavior_cloning` row to `training.jsonl` (`driver`, `agent_samples`, `final_epoch_mse`, `driver_failures`). "Driver failures" in the DAgger rounds show how often the student alone still crashes.

The reason cloning is only a warm start: behavior cloning copies the teacher, mistakes included. PPO then optimises the reward directly and can end up better than the teacher, as in the DAPG work (Rajeswaran et al., 2018).

### 5.5 Collecting experience: `train.Worker` and the rollout loop

- `Worker` owns one `SwarmEnv`. When an episode ends it starts a new one with a random scenario from the stage and a random training seed from 0–999.
- 8 workers step in lockstep. At every step the observations of 8 environments × 8 drones = 64 agents are stacked (`stack_teams`) and sent through the network in **one batch**. This is what "vectorized environments" means in the PPO implementation guides; more parallel environments also make updates less noisy.
- Each rollout is 256 steps per environment, so one update uses 256 × 8 × 8 = **16,384 drone transitions**.
- **Time limits vs failures**: a crash is *terminal* (no future value). Reaching the 24 s limit is only a *truncation*, so the value of the final observation is bootstrapped (Pardo et al., 2018). The loop keeps that final observation's value separately (`bootstrap`), because the worker has already reset to a new episode.
- `--failure-reset` (optional) restores a state saved about 2 s before a collision and continues with fresh on-policy steps. This is the replay idea from Huang et al. (ICRA 2024).

### 5.6 Advantages and reward scaling: `advantages`, `RewardScaler`

- `RewardScaler` divides rewards by a running standard deviation of the discounted return. This keeps value targets around 1 without changing which policy is best. Engstrom et al. (2020) identified this as one of the implementation details that most affect PPO. Yu et al. (2022) recommend value normalization for multi-agent PPO for the same reason.
- `advantages` computes GAE (Schulman et al., 2016) with γ = 0.99 and λ = 0.95:

```text
δ_t = r_t + γ·(1 − terminated_t)·V(s_{t+1}) − V(s_t)
A_t = δ_t + γ·λ·(1 − ended_t)·A_{t+1}          return target = A_t + V(s_t)
```

Advantages are then normalised to mean 0 and std 1 over the batch.

### 5.7 The PPO update (inside `train`)

For 5 epochs over the batch, in minibatches of 2,048:

```text
ratio   = exp(log π_new(a|s) − log π_old(a|s))
L_clip  = −mean( min(ratio·A, clip(ratio, 0.8, 1.2)·A) )         (Schulman et al., 2017)
L_value = 0.5·mean((V(s) − return)²)
loss    = L_clip + 0.5·L_value − 0.001·entropy
```

- Adam (eps 1×10⁻⁵); the learning rate is annealed linearly from 3×10⁻⁴ to 3×10⁻⁵ over each stage.
- Gradients are clipped to norm 0.5 **separately for actor and critic**.
- **Early stop**: if the mean approximate KL of an epoch exceeds 0.03, the remaining epochs are skipped, so one batch cannot push the policy too far.
- **Critic warm-up**: for the first 10 updates after cloning, and the first 5 updates of stages 2 and 3, only `L_value` is used and actor gradients are dropped. The critic learns what "good" looks like, including the new obstacle scenarios, before it steers the actor. These updates are logged with `"phase": "critic_warmup"`.
- After every update: one `ppo` row goes to `training.jsonl` and one line to the terminal, then `latest.pt` is saved (atomically, via a `.tmp` file).

### 5.8 Validation and the best checkpoint: `train.validate`, `evaluate.selection_score`

At the start, every 100,000 steps, and at the end, the **deterministic** policy (`tanh(mean)`, no noise) flies every stage scenario on selection seeds 5000–5004. Those seeds are never used for training or for the final test. `selection_score` turns the results into one number:

```text
score = success_rate − 0.25·min(slot_RMSE, 2 m) − 0.25·(1 − recovered_fraction)      (1.0 = perfect)
```

Survival dominates: one failed episode in five costs 0.2, the same as 0.8 m of slot error. If the score beats the best so far, the weights are saved as **`best.pt`**, and the full metrics as `best.json`. Every validation is appended to `evaluations.jsonl`.

Because stage 2 also validates at its start, the policy it inherits from stage 1 is its first candidate. **A stage's `best.pt` can therefore never be worse on validation than what the stage started from.** Keeping the best validated checkpoint instead of the last one is standard model selection. Train, validation and test seeds are kept separate in the way Cobbe et al. (2020) separate training and test levels.

### 5.9 Curriculum and the full pipeline: `src/swarmflow/curriculum.py`

- `STAGES` and `COMMON` hold the settings in section 4, and `stage_arguments(n, seed)` turns them into `train()` arguments. Curriculum learning, easy tasks first (Bengio et al., 2009; Narvekar et al., 2020), helps here because obstacle avoidance is useless until the formation itself is stable.
- `train_stage` trains one stage. It **skips** a stage that has `finished.json` and **continues** one that has `latest.pt` but no `finished.json`. So `run_pipeline`, which is `make training` and `swarmflow pipeline`, can simply be re-run after an interruption.
- `--continue` (`train(..., continue_run=True)`) reloads weights, optimizer, reward scale and the original settings, then trains only the remaining steps of the stage.
- With `--view` (`make run train`), `start_viewer` runs the viewer as a separate process, so it does not slow the training down. The viewer's live progress comes from `training_status`, which reads the newest stage folder's logs.

### 5.10 Showcase and the held-out test: `build_showcase`, `evaluate.learning_checks`

1. Take `best.json` from **every trained seed** of the final stage, including seeds trained in earlier sessions, and choose the run with the **highest validation score**.
2. Move any previous showcase to `models/showcase-archive/<date>/`, so no earlier result is lost. Then copy the chosen `best.pt` to `models/showcase/best.pt` and its training history (all stages) to `models/showcase/history/`.
3. Fly the chosen model **and** the predictive baseline on 8 scenarios × 20 **held-out seeds (100000–100019)**. These seeds are used only here, after the choice has been made.
4. `learning_checks` applies the acceptance targets from `docs/research-plan.md`:
   - no collisions or other failures;
   - at least 95% of disturbances recovered within 5 s;
   - slot RMSE below 0.35 m on hold, follow and turn.
5. Write `summary.json`, `SUMMARY.md` (including a per-stage table of steps and best validation scores) and the raw reports. **"Learned correctly: YES"** means every check passed.

### 5.11 Viewer: `server.py`, `web/`

- `python -m swarmflow view` (or `make run`, or the launcher) automatically loads `models/showcase/best.pt`. If there is no showcase yet, it loads the best stage checkpoint. It re-checks `models/` every 5 s and switches to a newer best model by itself. During training it therefore always flies the best model so far, and afterwards the final showcase model.
- The **Controller** menu switches between the learned policy, the predictive baseline and the slot-only ablation on the same scenario and seed, so they can be compared live.
- The **"Did the model learn correctly?"** panel (`/api/model`) shows:
  - while training runs, a progress bar with the stage, percentage, steps/s, time left and last validation score (it refreshes every 10 s);
  - the verdict and each acceptance check;
  - how long and how the model was trained, and which checkpoint was chosen;
  - three training curves: validation score (the ◆ marks the saved `best.pt`), training failure rate and training slot error, with the stages drawn one after another on a shared "PPO steps" axis;
  - a learned-vs-baseline table on the held-out seeds.

### 5.12 Files produced

```text
models/
  stage1-seed0/              one folder per stage and seed
    best.pt                  best validated weights (+ optimizer, reward scale, config)
    best.json                its validation metrics
    latest.pt                weights after the most recent update (used by --continue)
    checkpoints/step_*.pt    snapshots every 500,000 steps
    training.jsonl           one row per cloning round and per PPO update
    evaluations.jsonl        one row per validation
    config.json              every setting of the run
    finished.json            written when the stage completes
  stage2-seed0/ ...
  stage3-seed0/ ...
  showcase/                  what the viewer and the presentation use
    best.pt                  the selected model
    summary.json, SUMMARY.md verdict, per-stage table, checks, learned-vs-baseline table
    test_learned.json, test_baseline.json, test_compare.json
    validation.json          the selection metrics of the chosen checkpoint
    history/                 training logs of every stage that led to it
  showcase-archive/<date>/   earlier showcases, kept whenever a new one is built
```

`models/best.pt` and `models/latest.pt` directly inside `models/` were not produced by this code. The viewer never loads them.

---

## 6. Reading the training output

### 6.1 Terminal lines

```text
[clone] round 2 (student (DAgger)): 240,000 samples, mse 0.00412, driver failures 1/17
[eval:start] steps 0  score 0.912 (best 0.912)  success 100%  slot RMSE 0.183 m  -> new best, saved best.pt
[critic_warmup] 20,480/1,000,000 (2.0%)  reward/step +0.0371  failures(last 100) 3%  ...
[ppo] 409,600/1,000,000 (41.0%)  reward/step +0.0452  failures(last 100) 0%  slot RMSE(last 100) 0.12 m  kl 0.0081  value loss 0.214  602 steps/s  ETA 0:16:21
[eval:periodic] steps 400,000  score 0.975 (best 0.975)  ...  -> new best, saved best.pt
```

(Illustrative numbers.)

### 6.2 What healthy training looks like

- `failures(last 100)` falls and stays low. It will be higher at the start of stage 2, when obstacles first appear.
- `slot RMSE(last 100)` falls. It is measured *with* exploration noise, so it stays above the validation value.
- `reward/step` rises. Each drone can earn at most +0.05 per step.
- `kl` stays roughly between 0.001 and 0.03. A value near 0 means the policy is not changing, which was the old failure.
- `value loss` stays around 1 thanks to reward scaling, instead of hundreds.
- `explained_variance` (in the JSON log) climbs towards 1 as the critic learns.
- Validation `score` rises, and `best.pt` is replaced whenever it improves.

### 6.3 If a stage does not pass

Look at the curves in the viewer or in `evaluations.jsonl`:

- **The score is still rising at the end**: the budget was too small. Continue the stage for longer with `--resume models/stageN-seedS/best.pt --steps ... --output models/stageN-seedS-more`.
- **Many collisions early in stage 2**: try `--failure-reset`, or a lower `--init-log-std`.
- **Seeds differ a lot**: train more seeds (`make train-full-multi`). Results should be judged over several seeds (Henderson et al., 2018; Agarwal et al., 2021).

### 6.4 Pure reinforcement learning (no imitation)

`--clone-steps 0` skips cloning and starts PPO from random weights (with a larger initial std of 0.37). Pure PPO on an 8-drone formation with collisions needs far more steps, so the default keeps the cloning warm start. Even so, the part that makes the policy *good* is PPO, and the viewer's curves show how far it moved beyond the cloned start (the step-0 validation point).

---

## 7. What was used from which paper

| Paper | What we use |
| --- | --- |
| Schulman et al., *Proximal Policy Optimization Algorithms*, 2017 | Clipped surrogate objective (ε = 0.2), several epochs per batch. |
| Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR 2016 | GAE with γ = 0.99, λ = 0.95. |
| Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS 2022 | PPO with parameter sharing for cooperative agents, value normalization, few epochs, clip ≤ 0.2, large batches, optional centralized critic. |
| de Witt et al., *Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?*, 2020 | Independent PPO with a local critic as a strong default. |
| Andrychowicz et al., *What Matters in On-Policy Reinforcement Learning?*, ICLR 2021 | Separate policy and value networks, small final-layer init, lower initial action std, tanh squashing, Adam 3×10⁻⁴, γ as a key parameter. |
| Engstrom et al., *Implementation Matters in Deep Policy Gradients*, ICLR 2020 | Reward scaling, learning-rate annealing, orthogonal init and gradient clipping matter as much as the algorithm. |
| Huang et al., *The 37 Implementation Details of PPO*, ICLR Blog Track 2022 | Vectorized environments, Adam eps 1×10⁻⁵, advantage normalization, gradient clipping 0.5, KL monitoring. |
| Pardo et al., *Time Limits in Reinforcement Learning*, ICML 2018 | Bootstrap at time-limit truncation, never at failures. |
| Ross, Gordon & Bagnell, *DAgger*, AISTATS 2011 | Student-driven data collection labelled by the teacher, against covariate shift. |
| Laskey et al., *DART: Noise Injection for Robust Imitation Learning*, CoRL 2017 | Noise injected while the teacher flies, clean labels. |
| Rajeswaran et al., *Learning Complex Dexterous Manipulation with Deep RL and Demonstrations* (DAPG), RSS 2018 | Imitation warm start followed by RL fine-tuning. |
| Uchendu et al., *Jump-Start Reinforcement Learning*, ICML 2023; Nair et al., *AWAC*, 2020 | Warning that an imitation-initialized policy can be unlearned when fine-tuning starts with an untrained critic; hence critic warm-up and a separate critic. |
| Batra et al., *Decentralized Control of Quadrotor Swarms with End-to-end Deep RL*, CoRL 2021 | Shared local policy, relative neighbour observations, attention/deep-set encoders, PPO, large budgets (see `docs/research-plan.md`). |
| Huang et al., *Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep RL*, ICRA 2024 | Resets from pre-collision states (`--failure-reset`). |
| Lee et al., *Set Transformer*, ICML 2019; Zaheer et al., *Deep Sets*, NeurIPS 2017 | Permutation-invariant set pooling with attention. |
| Kaufmann et al., *Champion-level drone racing using deep RL*, Nature 2023 | Scale reference for simulated PPO training of a drone policy. |
| Bengio et al., *Curriculum Learning*, ICML 2009; Narvekar et al., JMLR 2020 | Staged curriculum from easy to hard scenarios. |
| Cobbe et al., *Leveraging Procedural Generation to Benchmark RL*, ICML 2020 | Separate training and test seeds to measure generalization. |
| Henderson et al., *Deep RL that Matters*, AAAI 2018; Agarwal et al., *Deep RL at the Edge of the Statistical Precipice*, NeurIPS 2021 | Several training seeds; do not trust a single run. |

These references were chosen from well-known work and from the six papers already analysed in `docs/research-plan.md`. Check exact figures (for example training-sample counts) in the papers themselves before quoting them on slides.

---

## 8. Limitations to state honestly

- The environment models **stabilized velocity-level flight**, not rotor thrust, attitude, downwash or real sensors (see the README). A good policy here is not a hardware-ready controller.
- The budgets in `STAGES` are starting budgets; whether they are enough is decided by the validation curves and the held-out verdict, not by the step count.
- The predictive baseline is a strong hand-coded controller. The learned policy matching it on held-out seeds is a good result; beating it is not guaranteed.
- `dense` (8–12 static spheres) and hills/canyon worlds are not in the training curriculum, so do not claim generalization to them without testing.

## 9. Presentation outline

1. **Problem**: 8 drones follow a target, make room for crossing objects and return to the *same* slots.
2. **Why RL**: learn a single shared local policy that works for any drone from local observations only.
3. **What went wrong before**, with the evidence in section 2: 84% crashes, 160/160 fly-aways, value loss in the thousands, KL ≈ 0.
4. **Fixes**: the table in section 3, each with its paper.
5. **Pipeline**: the diagram in section 4, including the time and the budget.
6. **Results**: open the viewer and show:
   - the model panel's verdict and curves;
   - the learned-vs-baseline table;
   - a live flight with the learned controller, then the same seed with the baseline.
7. **Limitations**: section 8.

---
---

# Part B: learning guide (understand the approach from scratch)

Sections 1–9 above are the reference: the evidence, the fixes and the code. This part is for **reading and understanding**. It explains our new approach in plain language, the reinforcement-learning ideas behind it (always with our drones as the example), what one training run actually does, what we do and don't do, and where each idea comes from. Read it top to bottom once, and sections 1–9 will make much more sense afterwards.

## 10. Our new approach in one page

**The task.** Eight drones follow a moving target in a fixed formation. When an object comes through, they make room, then return to **their own** slots. One neural network, shared by all drones, decides each drone's velocity 20 times per second, using only what that drone can see nearby.

**How we train it, in six steps:**

1. **Copy a teacher first (behavior cloning).** Our own hand-coded predictive controller flies, and the network learns to imitate its commands. We add noise while the teacher flies (DART), and then let the student fly while the teacher corrects it (DAgger), so the network also learns how to recover from mistakes. Result: a reasonable starting pilot after about 6 minutes.
2. **Teach the judge before the pilot moves (critic warm-up).** For a few updates only the critic learns, so its feedback makes sense before it starts changing the pilot.
3. **Improve by trial and error (PPO).**
   - Eight simulated worlds run at once, and every drone gets a score (reward) at every step:
     - **+1 per second alive**;
     - minus penalties for slot error, speed error, getting too close, and jerky commands;
     - **−100** if the team crashes.
   - PPO nudges the network towards actions that turned out better than the critic expected, and away from worse ones, in small, safe steps.
4. **Easy before hard (curriculum).** Stage 1 is formation only, stage 2 adds moving objects, and stage 3 is the full benchmark. Each stage starts from the best model of the previous one.
5. **Keep the best, not the last.** Every 100,000 steps the model is tested on episodes it never trained on; if it scores higher than before, it is saved as `best.pt`.
6. **Final exam on unseen episodes (showcase).** The best model and the hand-coded baseline fly the same 160 never-seen episodes. The acceptance checks decide the verdict, and the viewer shows it.

**Old vs new, side by side:**

| | Old training | New training |
| --- | --- | --- |
| Budget | 150,000 steps (about 2–5 min) | 5,000,000 steps in 3 stages (about 3 h) |
| Worlds per update | 1 | 8 in parallel, 16,384 drone transitions per update |
| Critic | shared with the actor, and it swamped the actor's updates | separate network, clipped separately, warmed up first |
| Reward | always negative, unbounded, −200 crash, so crashing on purpose paid off | +1/s alive, capped costs, −100 crash, so staying alive always pays off |
| Imitation | copied only perfect teacher states | teacher with noise (DART) + student-driven rounds (DAgger) |
| Exploration noise | ±1.1 m/s, so 84% of episodes crashed | ±0.6 m/s at the start, learned afterwards |
| Model kept | the last one, overwritten after every update | the best one on validation seeds (`best.pt`) |
| Evaluation | same seeds for tuning and reporting | separate seeds for training, selection and final test |
| Viewer | hand-coded controller only | learned model by default, live training view, verdict panel |

## 11. Background concepts, with our drones as the example

### 11.1 The reinforcement-learning loop

| Term | Meaning in general | In SwarmFlow |
| --- | --- | --- |
| **Environment** | The world the agent acts in | `SwarmEnv`: drones, target, objects, flight physics, collision checks |
| **Agent** | The decision maker | The controller of one drone; all 8 drones run copies of the same network |
| **Step** | One tick of the loop | 0.05 s of simulated time (20 Hz) |
| **Episode** | One attempt from start to end | 24 s (480 steps), or less if the team fails |
| **Observation** | What the agent sees | Its vector to its own slot, its velocity, the target's velocity, its last command, slot offset, altitude and time left; plus the 6 nearest drones and 6 nearest objects within 7 m (relative position, relative velocity, radius) |
| **Action** | What the agent does | A desired velocity: 3 numbers in [−1, 1], scaled to at most 3 m/s |
| **Reward** | A score for that moment | Section 5.2: +1/s alive minus penalties; −100 on team failure |

The loop is: observe → act → the world moves → reward → observe again. **Training means changing the network so that the total reward over time gets larger.** Nobody tells the drone the "right" action; it discovers which actions lead to higher reward.

### 11.2 Policy

The **policy** is the network that turns an observation into an action. Ours outputs a mean velocity and a spread (standard deviation).

- During **training**, the action is **sampled** around the mean, so the drone tries slightly different things (exploration).
- During **evaluation and in the viewer**, it uses the **mean** only (deterministic), so the same situation always gives the same action.

### 11.3 Return and discount factor

The **return** is the reward added up over the future, with later rewards counting a little less:

```text
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …      with γ = 0.99
```

γ (gamma) is the **discount factor**. A useful rule of thumb is the "effective horizon" 1/(1−γ) = 100 steps, which is **about 5 seconds** for us. The drone therefore cares about what happens over roughly the next 5 s, which matches our recovery target (back in formation within 5 s).

### 11.4 Value function and the critic

The **value** V(o) is the return the drone can *expect* from its current situation if it keeps following the current policy. The **critic** is a second network that learns to predict this number from the observation.

- The critic never flies; it only **judges**.
- It is trained like ordinary regression: predict the return, compare with what actually happened, and reduce the error (the "value loss").

### 11.5 Advantage: "better or worse than expected?"

```text
advantage = what actually happened − what the critic expected
one-step form: A ≈ r + γ·V(o_next) − V(o)
```

Example:

- The critic expects a return of +3.0 when an object approaches. The drone dodges early and is back in its slot quickly, and the actual return is +3.4. The advantage is **+0.4**, so that action becomes a little more likely.
- If the drone had drifted away and scored +2.1, the advantage would be **−0.9**, and that action would become less likely.

Advantages turn a single reward number into a learning signal for **each individual action**.

### 11.6 From policy gradient to PPO

- The basic **policy-gradient** idea: raise the probability of actions with positive advantage, and lower it for actions with negative advantage.
- **The danger:** one large update can wreck a policy that took hours to learn.
- **PPO (Proximal Policy Optimization)** makes updates safe by comparing the new policy with the old one on the same actions:

```text
ratio = π_new(a|o) / π_old(a|o)
objective = mean( min( ratio·A , clip(ratio, 0.8, 1.2)·A ) )
```

In words: **no single update is rewarded for changing an action's probability by more than about 20%**. Because each update is limited, the same batch of experience can safely be reused for several passes (epochs; we use 5). We also stop a batch early if the policy has already moved too far, measured by the KL divergence (limit 0.03).

### 11.7 GAE: a better advantage estimate

There are two ways to estimate an advantage:

- The **one-step** estimate trusts the critic a lot. It is low-noise, but wrong whenever the critic is wrong.
- The **full return** trusts only real rewards. It is unbiased but very noisy.

**GAE (Generalized Advantage Estimation)** blends the two with a parameter λ (we use 0.95), roughly "mostly real rewards, with the critic filling in the far future". The formula is in section 5.6.

### 11.8 On-policy learning: why RL needs so many steps

PPO is **on-policy**: it learns only from flights made by the *current* network. After each update, the batch is thrown away and new flights are collected. That is the main reason RL needs millions of steps:

- Our 5,000,000 steps are 250,000 s of simulated time, about **69 hours of simulated flight** per seed, or 40 million drone decisions.
- A fast simulator and 8 parallel worlds compress this into about 3 hours.

### 11.9 Exploration: noise, standard deviation, entropy, tanh

- **Noise**: during training the action is `mean + noise`. The noise's standard deviation is a learned parameter (`log_std`).
- **Starting std**: 0.2 after cloning, about ±0.6 m/s. It is kept small because neighbouring drones are only 0.89 m apart surface to surface; the old ±1.1 m/s crashed 84% of episodes.
- **Entropy bonus** (0.001): a small reward for keeping some randomness, so exploration does not collapse to zero too early.
- **tanh squashing**: the sampled value is passed through tanh so the command always stays inside [−1, 1]. The probability calculation is corrected for this squashing (`log_probability`).

### 11.10 Actor–critic: why a separate critic

The **actor** is the policy (it flies) and the **critic** is the value function (it judges). They can share one network body to save parameters. The old code did that, and the critic's huge errors took over the shared body and the gradient budget, so the pilot barely learned. We now use **two separate networks** of about 72k parameters each, as Andrychowicz et al. (2021) recommend for continuous control.

### 11.11 Multi-agent RL: parameter sharing, IPPO, MAPPO, CTDE

- **Parameter sharing**:
  - all 8 drones use **one** network, so every step gives 8 training examples;
  - all drones behave consistently;
  - the input is local, so the same network can run on any number of drones.
- **Independent PPO (IPPO)**: each drone's critic sees only that drone's observation. This is our default, and it is simple and strong (de Witt et al., 2020).
- **MAPPO**: the critic also sees team information, during training only; the actors stay local. This general idea is called **centralized training, decentralized execution (CTDE)** (Lowe et al., 2017). `--critic central` gives a simple version (the mean of the team's features). Yu et al. (2022) showed that PPO in this cooperative setting is a strong baseline.
- **Team incentive**: the −100 failure penalty is given to *every* drone when *any* drone crashes, so each drone learns to avoid causing trouble for the others.

### 11.12 Seeing neighbours as a set: Deep Sets and attention

A drone may see 0 to 6 neighbours, listed in any order. An ordinary network would give different answers for "A then B" and "B then A".

- **Deep Sets** (Zaheer et al., 2017): embed each neighbour separately, then combine them with an order-independent operation.
- **What we use** (`SetEncoder`): a learned **query** attends over all neighbours (attention pooling, from the Set Transformer; Lee et al., 2019), with empty rows masked out. The drone can therefore focus on the most important neighbour or object, such as the closest one on a collision course.
- The unit test `test_attention_encoder_permutation_invariance` checks that the order truly does not matter.

### 11.13 Imitation learning: behavior cloning, covariate shift, DART, DAgger

- **Behavior cloning (BC)** is ordinary supervised learning on pairs of (observation, teacher action).
- **The problem, covariate shift.** The student is never perfect. A small error puts it in a situation the teacher never visited, where there is no training data, so it makes a bigger error, and so on. Ross et al. (2011) show that with plain BC the total error can grow roughly with the **square** of the episode length.
- **DART** (Laskey et al., 2017): the teacher flies with **noise added**, while the labels stay the teacher's clean commands. The data therefore contains "slightly off" situations together with the correct way back.
- **DAgger** (Ross et al., 2011): the **student flies** and the **teacher labels** every state it reaches, and the network retrains on all the data so far. The data comes exactly from the situations the student really gets into, and the error grows only about **linearly**.
- **Our teacher** is `PredictiveController`, our own code. At every step it tries many candidate velocities, predicts 2 s ahead, and picks the one with the lowest cost. It is good but slow: about 3.6 ms per 8-drone step on this Mac.

### 11.14 Warm start plus fine-tuning: why the critic must be ready

- After cloning, the actor can fly but the critic knows nothing. Its advantages are noise, so the first PPO updates can erase the cloned skill. Uchendu et al. (2023, Jump-Start RL) and Nair et al. (2020, AWAC) describe this kind of early "unlearning" when fine-tuning an imitation-initialized policy.
- **Our remedy**:
  - a separate critic;
  - critic-only warm-up updates (10 after cloning, 5 at the start of stages 2 and 3);
  - small initial exploration noise.
- **Why warm start at all?** From random weights, the drones would spend a long time crashing into each other before they discover formation flying. The DAPG work (Rajeswaran et al., 2018) showed that imitation followed by RL fine-tuning learns much faster than RL alone, and can end up better than the demonstrations.

### 11.15 Reward design: shaping and the "end it early" trap

- **Reward shaping** means giving feedback at every step (slot error, speed error, closeness) instead of only "success" or "fail" at the end. It makes learning much faster.
- **The trap.** If every step's reward is negative and a failure ends the episode, ending the episode can look attractive to the agent.
  - In the old reward, a drone 7 m from its slot lost about 2.45 per step, and the discounted future (×100) was about −245. A crash cost only −200.
  - So crashing was "cheaper", and the stage 2 model flew out of the world in 160 of 160 episodes.
- **The fix**, the same idea as the "healthy reward" in the MuJoCo Hopper benchmark:
  - **+1 per second alive**;
  - **capped** costs;
  - a **−100** crash penalty.

  The worst possible way of staying alive is worth about −60, which is still better than −100, so the agent never wants to end the episode.
- Example rewards per drone per step:

| Slot error | Reward |
| --- | --- |
| 0.1 m | +0.0495 (maximum is +0.05) |
| 0.5 m | +0.0375 |
| 2 m | −0.10 |
| Crash | −100 |

### 11.16 The small tricks that make PPO work (and why)

Engstrom et al. (2020) and Andrychowicz et al. (2021) showed that these "details" often matter as much as the algorithm itself. The 37-details guide (Huang et al., 2022) lists them.

| Trick | Why it helps | Our setting |
| --- | --- | --- |
| Parallel environments | More varied data per update, so less noisy learning | 8 worlds |
| Advantage normalization | Keeps every update a similar size | mean 0, std 1 per batch |
| Reward scaling | Keeps the critic's targets around 1 instead of hundreds | running std of the discounted return |
| Gradient clipping | One rare huge gradient cannot wreck the weights | norm 0.5, actor and critic separately |
| Learning-rate annealing | Big steps early, fine steps at the end | 3×10⁻⁴ → 3×10⁻⁵ |
| Small last-layer init | The first actions are near zero, not wild | orthogonal, gain 0.01 |
| KL early stop | Stops a batch that is already moving the policy too far | 0.03 |
| Time-limit bootstrapping | Reaching 24 s is not a failure, so the future still counts; a crash is truly the end | Pardo et al. (2018) |

### 11.17 Curriculum learning

- **Why**: learning to dodge objects is pointless before the swarm can hold its formation. Starting with everything at once would mostly produce crashes and very little useful signal.
- **How**: each of our 3 stages adds harder scenarios **and keeps the easier ones**, so earlier skills are not forgotten. Each stage starts from the previous stage's best model (Bengio et al., 2009; Narvekar et al., 2020).

### 11.18 Measuring honestly: seed splits, deterministic evaluation, several seeds

A **seed** fixes the random details of an episode: exact start positions, when and where an object comes, its speed and size. We keep four separate seed ranges:

| Seeds | Used for | Why separate |
| --- | --- | --- |
| 0–999 | Training episodes | The model learns from these |
| 5000+ | Choosing `best.pt` during training | Selection must not use training episodes |
| 1000+ | Our own checks (`make eval-stageN`, tests 9–11) | Experimenting without touching the final test |
| 100000+ | The final showcase test only | Used once, after the choice, so the reported result is not tuned to it |

This is the same idea as a train / validation / test split in ordinary machine learning; Procgen (Cobbe et al., 2020) uses separate training and test levels for the same reason.

- **Deterministic evaluation** (no noise) is how the viewer flies, so the verdict describes what you see.
- **Several training seeds**, meaning several independent runs, matter because RL results vary from run to run (Henderson et al., 2018; Agarwal et al., 2021).
- **Comparing with the baseline on the same seeds** gives the result a meaning: "as good as the hand-coded planner" is a clear statement.

## 12. One full training run, step by step

What actually happens when you type `caffeinate -i make run train`:

1. The Makefile runs `python -m swarmflow pipeline --seeds 0 --view`.
2. `start_viewer` starts the viewer as a separate process on port 8765 and opens the browser. The viewer watches `models/` from then on.
3. `run_pipeline` prints the plan (stages, steps, where results go).
4. **Stage 1** (`train_stage(1)` → `train(...)`), about 35–45 min:
   1. **Behavior cloning.** Round 0: the teacher flies 10,000 steps with noise, giving 80,000 labelled samples, and the network trains on them for 15 epochs. Rounds 1–3 (DAgger): the student flies 10,000 steps each, and the teacher labels. That is 320,000 samples in total, taking about 5–7 min.
   2. **First validation** on 3 scenarios × 5 selection seeds gives the first score and the first `best.pt`. Within about 5 s the viewer switches to this model.
   3. **PPO loop**, about 489 updates:
      - each update collects 8 worlds × 256 steps = 2,048 environment steps (16,384 drone transitions);
      - it computes GAE advantages, then runs up to 5 epochs × 8 minibatches;
      - it saves `latest.pt` and prints one line.
      - The first 10 updates train only the critic.
   4. **Every 100,000 steps**: validation. If the score improved, `best.pt` and `best.json` are replaced, and the viewer picks up the new model.
   5. At 1,000,000 steps: the final validation, then `finished.json`.
5. **Stage 2** (about 1 h): starts from stage 1's `best.pt`, validates it on 5 scenarios as its first candidate, runs 5 critic-only updates, then about 977 PPO updates (2,000,000 steps).
6. **Stage 3** (about 1 h): the same, from stage 2's `best.pt`, on all 8 scenarios.
7. **Showcase** (about 7–10 min):
   - chooses the best stage-3 run by validation score and copies it to `models/showcase/`;
   - tests it and the baseline on 8 scenarios × 20 unseen seeds;
   - runs the acceptance checks and writes `SUMMARY.md`.

   The viewer switches to `models/showcase/best.pt` and shows the verdict.
8. The viewer stays open until Ctrl+C.

If anything stops the run, the same command continues it: finished stages are skipped, and the current stage resumes from `latest.pt`.

## 13. What we do and what we don't do

**We do:**

- Train the network **ourselves**, from random weights, on this laptop.
- Warm-start by imitating **our own** hand-coded controller, then optimize the reward with PPO.
- Give the policy only **local, relative** observations, the kind of thing a drone could sense nearby.
- Share **one** policy across all drones.
- Command **velocities** to a stabilized flight model.
- Train with a curriculum, keep the best validated checkpoint, test once on unseen seeds, and compare with the baseline.
- Show the learned policy in the viewer with **nothing** running behind it.

**We don't:**

- Use any **pre-trained** model or external weights.
- Give the policy **global** information, such as every drone's position or the target's future path.
- Simulate rotors, thrust, attitude, downwash, real sensors or hardware. That is future work, for example with gym-pybullet-drones (Panerati et al., 2021).
- Put a **safety filter** or the baseline behind the learned policy.
- **Train inside the viewer.** The viewer only runs saved weights.
- Use **memory** (no recurrent network). Each decision depends on the current observation only.
- Promote stages **automatically** by performance. The stage budgets are fixed.
- **Tune on test seeds**, or claim formal safety guarantees.
- Train on the dense-sphere scenario or on the hills, canyon or pillars worlds, so we don't claim they work without testing.

## 14. Concept → code map

| Concept | Where in the code |
| --- | --- |
| Environment, observations, actions, collisions | `env.py`: `SwarmEnv.step`, `_observe`, `_clearance` |
| Reward | `env.py`: `SwarmEnv._reward` |
| Policy and critic networks | `policy.py`: `SetEncoder`, `Encoder`, `ActorCritic` |
| Deterministic flying (viewer, evaluation) | `policy.py`: `ActorCritic.act`, `ModelController`, `LearnedController` |
| Teacher and ablation controllers | `baseline.py`: `PredictiveController`, `SlotController` |
| Behavior cloning, DART, DAgger | `train.py`: `clone`, `_fit_clone` |
| Parallel rollouts | `train.py`: `Worker` and the rollout loop in `train` |
| Reward scaling | `train.py`: `RewardScaler` |
| GAE | `train.py`: `advantages` |
| PPO update, critic warm-up, KL stop, learning-rate annealing | `train.py`: the update loop in `train` |
| Validation and `best.pt` | `train.py`: `validate`; `evaluate.py`: `selection_score` |
| Curriculum and resumable pipeline | `curriculum.py`: `STAGES`, `COMMON`, `train_stage`, `run_pipeline` |
| Final test and verdict | `curriculum.py`: `build_showcase`; `evaluate.py`: `learning_checks` |
| Live training view and model panel | `curriculum.py`: `training_status`, `model_report`; `server.py`: `make_server`; `web/app.js` |
| Commands | `Makefile`; `cli.py` |
| Tests | `tests/test_learning.py`, `tests/test_showcase.py`, `tests/test_environment.py` |

## 15. Numbers to remember

- **Swarm and flight**:
  - 8 drones, 20 decisions per second, 24 s episodes (480 steps);
  - 1.25 m spacing (0.89 m surface gap);
  - maximum speed 3 m/s.
- **Observation**: 17 own values, plus 6 neighbours and 6 objects (8 values each) within 7 m.
- **Network**: about 72k actor parameters and 72k critic parameters.
- **Reward**: at most +0.05 per drone per step; −100 on a crash. γ = 0.99 (about a 5 s horizon), λ = 0.95.
- **Cloning**: 4 rounds × 10,000 steps = 320,000 labelled samples.
- **PPO**:
  - 16,384 drone transitions per update (8 worlds × 256 steps × 8 drones);
  - 5 epochs, minibatch 2,048, clip 0.2;
  - learning rate 3×10⁻⁴ → 3×10⁻⁵, KL limit 0.03.
- **Budget**:
  - 1M + 2M + 2M = 5M environment steps = 40M drone transitions;
  - about 69 h of simulated flight, about 3 h of real time.
- **Validation and test**: validation every 100,000 steps; final test on 8 scenarios × 20 unseen seeds.
- **Pass**: no failures, at least 95% of disturbances recovered within 5 s, slot RMSE below 0.35 m on hold, follow and turn.
- **The old runs**: 150,000 steps, 84% of training episodes crashed, 160 of 160 evaluation episodes flew out of the world.

## 16. Glossary

| Term | Meaning |
| --- | --- |
| Ablation | A deliberately weakened version used for comparison (slot-only: no avoidance) |
| Actor / critic | The network that acts / the network that predicts expected return |
| Advantage | How much better an action turned out than the critic expected |
| Baseline | The hand-coded predictive controller we compare against |
| Behavior cloning (BC) | Supervised learning that copies a teacher's actions |
| Checkpoint | A saved file of network weights (+ optimizer state and settings): `best.pt`, `latest.pt` |
| CTDE | Centralized training, decentralized execution |
| Curriculum | Training on easy tasks first, then harder ones |
| DAgger | Imitation where the student flies and the teacher labels the states it reaches |
| DART | Imitation where the teacher flies with added noise and clean labels |
| Deterministic policy | Always takes the mean action (no noise); used for evaluation and viewing |
| Entropy | How random the policy is; a small bonus keeps some exploration |
| Epoch / minibatch | One pass over the collected batch / a slice of it used for one gradient step |
| Episode | One 24 s flight (or shorter if it fails) |
| Explained variance | How well the critic predicts returns (1 = perfect, 0 = useless) |
| GAE | Generalized Advantage Estimation, which blends short- and long-term advantage estimates |
| Held-out test | Evaluation on seeds never used for training or model selection |
| IPPO / MAPPO | Independent PPO (local critic) / multi-agent PPO with a centralized critic |
| KL divergence | How much the policy changed in one update |
| Parameter sharing | All agents use the same network weights |
| PPO | Proximal Policy Optimization, a policy-gradient method with clipped, safe updates |
| Recovery | All drones back within 0.35 m of their own slots (and slow) for 1 s after an object leaves |
| Return | Discounted sum of future rewards |
| Rollout | The batch of experience collected before one update |
| Seed | The number that fixes an episode's random details |
| Slot RMSE | Root-mean-square distance between drones and their assigned slots over an episode |
| Termination vs truncation | Episode ended by failure (no future) vs by the 24 s time limit (the future still counts) |
| Validation score | Our single number for choosing `best.pt` (section 5.8) |

## 17. Reading list, in a sensible order

Start with 1–3 for the basics. Then read the parts that match what you want to explain in the presentation.

1. **OpenAI Spinning Up**, "Introduction to RL", parts 1–3, and the PPO page: the clearest short introduction to policies, returns, value functions, advantages and PPO. <https://spinningup.openai.com/en/latest/>
2. **Sutton & Barto, *Reinforcement Learning: An Introduction*** (2nd ed., 2018), chapters 1, 3 and 13 (the book is free online): the standard textbook for the concepts in 11.1–11.6. <http://incompleteideas.net/book/the-book-2nd.html>
3. **Schulman et al., *Proximal Policy Optimization Algorithms*** (2017), sections 1–3: the clipped objective we use. <https://arxiv.org/abs/1707.06347>
4. **Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*** (ICLR 2016), the GAE definition. <https://arxiv.org/abs/1506.02438>
5. **Huang et al., *The 37 Implementation Details of PPO*** (ICLR Blog Track 2022): every practical trick in 11.16, with code. <https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/>
6. **Andrychowicz et al., *What Matters in On-Policy Reinforcement Learning?*** (ICLR 2021): read the recommendations boxes (separate networks, initialization, action std, γ). <https://arxiv.org/abs/2006.05990>
7. **Engstrom et al., *Implementation Matters in Deep Policy Gradients*** (ICLR 2020): why reward scaling, learning-rate annealing and similar details matter. <https://arxiv.org/abs/2005.12729>
8. **Pardo et al., *Time Limits in Reinforcement Learning*** (ICML 2018): termination vs truncation. <https://arxiv.org/abs/1712.00378>
9. **Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*** (NeurIPS 2022): PPO for teams of agents, and its practical suggestions. <https://arxiv.org/abs/2103.01955>
10. **de Witt et al., *Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?*** (2020): independent PPO as a strong default. <https://arxiv.org/abs/2011.09533>
11. **Lowe et al., *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*** (MADDPG, NeurIPS 2017): the CTDE idea behind centralized critics. <https://arxiv.org/abs/1706.02275>
12. **Ross, Gordon & Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*** (DAgger, AISTATS 2011). <https://arxiv.org/abs/1011.0686>
13. **Laskey et al., *DART: Noise Injection for Robust Imitation Learning*** (CoRL 2017). <https://arxiv.org/abs/1703.09327>
14. **Rajeswaran et al., *Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations*** (DAPG, RSS 2018): imitation warm start + RL. <https://arxiv.org/abs/1709.10087>
15. **Uchendu et al., *Jump-Start Reinforcement Learning*** (ICML 2023), and **Nair et al., *AWAC*** (2020): pitfalls when fine-tuning an imitation-initialized policy. <https://arxiv.org/abs/2204.02372>, <https://arxiv.org/abs/2006.09359>
16. **Zaheer et al., *Deep Sets*** (NeurIPS 2017), and **Lee et al., *Set Transformer*** (ICML 2019): order-independent set encoders. <https://arxiv.org/abs/1703.06114>, <https://arxiv.org/abs/1810.00825>
17. **Batra et al., *Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning*** (CoRL 2021): the closest prior work to our project (read with `docs/research-plan.md`). <https://arxiv.org/abs/2109.07735>
18. **Huang et al., *Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep Reinforcement Learning*** (ICRA 2024): pre-collision resets (`--failure-reset`). <https://arxiv.org/abs/2309.13285>
19. **Kaufmann et al., *Champion-level drone racing using deep reinforcement learning*** (Nature 2023): PPO-trained drone policy, a sense of scale. <https://www.nature.com/articles/s41586-023-06419-4>
20. **Bengio et al., *Curriculum Learning*** (ICML 2009) and **Narvekar et al., *Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey*** (JMLR 2020). <https://dl.acm.org/doi/10.1145/1553374.1553380>, <https://arxiv.org/abs/2003.04960>
21. **Cobbe et al., *Leveraging Procedural Generation to Benchmark Reinforcement Learning*** (ICML 2020): separate train and test episodes. <https://arxiv.org/abs/1912.01588>
22. **Henderson et al., *Deep Reinforcement Learning that Matters*** (AAAI 2018) and **Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice*** (NeurIPS 2021): why several seeds and careful reporting. <https://arxiv.org/abs/1709.06560>, <https://arxiv.org/abs/2108.13264>
23. **Gymnasium Hopper documentation**, the "healthy reward" behind our alive bonus: <https://gymnasium.farama.org/environments/mujoco/hopper/>. **PettingZoo documentation**, the multi-agent API our environment follows: <https://pettingzoo.farama.org/>
24. **Panerati et al., *Learning to Fly*** (gym-pybullet-drones, IROS 2021): the next step towards full quadrotor physics. <https://arxiv.org/abs/2103.02142>

These links were written from memory without internet access in this session; if one does not open, search for the title.

## 18. Questions you may be asked (with answers)

**Did you use a pre-trained model?** No. The network starts from random weights and is trained on our machine. It first imitates our own hand-coded controller, then improves with PPO. No external weights are used anywhere.

**Isn't imitating the hand-coded controller cheating?** It is only the starting point. The final policy is shaped by PPO optimizing the reward, and the curves in the viewer show how the validation score moves after cloning (the step-0 point). The teacher is our own code, and `--clone-steps 0` trains with pure RL if needed.

**Why use RL at all, if the hand-coded controller already works?** The hand-coded planner is our *reference*. It is hand-tuned and runs an optimization at every step (about 3.6 ms per 8-drone step here). The research question is whether a learned, local, shared policy can reach the same behaviour directly from a reward signal. Such a policy is cheap to run and can later be trained on things that are hard to hand-design, such as noisy sensing or richer dynamics.

**Why PPO and not DQN or SAC?**

- DQN needs a small set of discrete actions, while ours are continuous velocities.
- PPO is stable, works naturally with many parallel environments and with parameter sharing, and is the standard choice in cooperative multi-agent RL (MAPPO) and in drone work (Batra et al.; Kaufmann et al.).
- Off-policy methods such as SAC reuse old data more efficiently. But our simulator is cheap, and in multi-agent settings old data goes stale as the other agents' behaviour changes.

**Why did the old training take 3 minutes and the new one about 3 hours?** The old budget was 150,000 steps, and the setup was broken: 84% crashes, 160/160 fly-aways, and a policy that barely changed (section 2). The new budget is 33 times larger (5,000,000 steps), and the failures behind the broken runs are fixed (section 3).

**How do you know the model has really learned?** Not from the training reward. The evidence is:

- the **held-out test** on 160 unseen episodes;
- the **acceptance checks** (no failures, at least 95% recovered within 5 s, slot error below 0.35 m);
- the **comparison with the baseline** on the same seeds;
- the **validation curve**.

All of these are shown in the viewer and in `models/showcase/SUMMARY.md`.

**Why keep `best.pt` instead of the final weights?** RL performance goes up and down during training. Keeping the checkpoint with the best score on separate validation seeds is standard model selection. The final test then uses yet another set of seeds, so the choice cannot flatter the reported result.

**Why a separate critic?** In the old shared network, the critic's large errors dominated the training signal, so the pilot barely changed (KL ≈ 0.0005). With separate networks, each learns its own job.

**Why not a centralized critic (MAPPO)?** It is available (`--critic central`). Independent PPO with a local critic is simpler and has been shown to be strong. A centralized critic is a controlled comparison you can run as an experiment.

**Why three stages?** Dodging objects only makes sense once the formation is stable. Each stage keeps the easier scenarios so they are not forgotten, and starts from the previous stage's best model.

**What is a "step"?** One 0.05 s tick of the whole swarm. 5,000,000 steps are about 69 hours of simulated flight, or 40 million individual drone decisions.

**Would it work with 16 drones?** The network is size-independent, because neighbours are handled as a set, and the viewer lets you change the drone count. But training used 8 drones, so test it first (for example `python -m swarmflow eval --policy learned --checkpoint models/showcase/best.pt --drones 16`) before claiming it.

**Is it ready for real drones?** No. The simulator models stabilized velocity-level flight, with no rotors, attitude, downwash or real sensors, and there is no formal safety guarantee. A physics-based validation (for example gym-pybullet-drones) would be the next step.

**What if the verdict says NOT LEARNED YET?** Look at which check failed and at the curves, then follow section 6.3: continue the stage longer, use `--failure-reset` for collision-heavy stages, or train more seeds.

---
---

# Part C: training version 2 (after the first full run) — current

## 19. What the first full run showed

Numbers from `training_logs.txt` and `models/` of the first complete run (seed 0):

| Stage | Result |
| --- | --- |
| **1** (hold, follow, turn) | **Worked.** Validation score 0.997, 0 collisions in 2,080 training episodes, slot RMSE about 0.01 m. |
| **2** (+ crossing, head-on) | Started at 0.345: crossing and head-on 0/5 each, because the network had never been taught to avoid anything. It peaked at 0.788 after 100k steps, then **fell to 0.2–0.3** for most of the stage. 20–50% of training episodes failed, and it ended at 0.196. |
| **3** (all 8 scenarios) | Started from stage 2's early best. Best score 0.496 (at its start); double crossing 0% success, slalom slot error about 2.6 m. |
| **Showcase** | 138/160 episodes without failure and 38/80 disturbances recovered, while the baseline reached 160/160 and 80/80. Slot error when undisturbed passed (under 0.35 m). Verdict: **NOT LEARNED YET**. |

## 20. Why stages 2 and 3 failed

1. **The network never saw an obstacle being avoided.** Behavior cloning ran only in stage 1, on hold, follow and turn. The teacher dodges perfectly (100% on every obstacle scenario), but that knowledge was never passed on.
2. **Exploration collapsed.** The action noise shrank from 0.20 to 0.015 during stage 1, which was fine for precise formation flying. Stages 2 and 3 started with it, and it stayed around 0.035–0.04 (about ±0.1 m/s). A dodge needs a 1–2 m sideways move, which that much noise almost never produces by chance.
3. **Blame was shared.** When any drone crashed, all 8 got −100. Seven of them had done nothing wrong, so the learning signal was mostly noise.
4. **The warning came too late.** The obstacle closeness penalty started only 0.45 m from contact and was tiny next to −100, so the drones mostly learned about objects by hitting them.
5. **Nothing held on to good behaviour.** PPO drifted away from the skill it found early. Selection used only 5 seeds per scenario, so scores jumped around (0.79 → 0.26 → 0.40 → 0.23).

## 21. What version 2 changes, and where each idea comes from

| # | Change | Why | Source | Code |
| --- | --- | --- | --- | --- |
| 1 | **Behavior cloning + DAgger in every stage**, on that stage's own scenarios. Stage 1 uses 4 rounds × 10,000 steps; stages 2 and 3 use 4 rounds × 20,000 steps, starting from the previous stage's best model. | The student learns obstacle avoidance from the teacher on the states it actually reaches. | Ross et al. 2011 (DAgger); Laskey et al. 2017 (DART) | `train.clone`, `train._collect`, `curriculum.STAGES` |
| 2 | **Imitation loss inside PPO.** Each PPO minibatch adds the squared error to the teacher on 1,024 samples of the stage's teacher data. Its weight falls linearly from 1.0 to 0.1 over each stage. | PPO can improve on the teacher but cannot drift far from it; the pull weakens as training goes on. | Rajeswaran et al. 2018 (DAPG: demonstration term in the policy gradient, decaying weight); Schmitt et al. 2018 (Kickstarting: teacher loss with decaying weight) | `train` update loop (`bc_coef`, `bc_coef_end`) |
| 3 | **Exploration that cannot collapse.** The noise std has a floor of 0.05 (`min_log_std = −3`), is reset to 0.2 at the start of each new stage, and there is no entropy bonus. | Enough noise to discover dodges, and a fresh start for each new skill. | PPO paper MuJoCo setting and the quadrotor-swarm training setup: no entropy bonus. Andrychowicz et al. 2021: the initial std matters. | `policy.ActorCritic(min_log_std)`, `train` (std reset) |
| 4 | **Crash penalty only for the drones involved** (−100); the episode still ends for all. | Each drone is blamed for its own collisions. The others still lose their future reward because the episode ends. | Batra et al. 2021: individual collision penalties | `env.SwarmEnv._reward`, `culprits` |
| 5 | **Obstacle warning zone.** A cost of 3 × (1 − gap / 1 m)² per second, capped at the value of two obstacles. | The drones feel an object coming about 1 m before contact and move away early. | Batra et al. 2021: smooth collision penalty with a fall-off radius | `env.OBSTACLE_MARGIN_M`, `OBSTACLE_WEIGHT` |
| 6 | **Pre-collision replay** (`--failure-reset`) switched on in stages 2 and 3. | After a crash, the drones retry the same dangerous moment from about 2 s earlier, so rare hard situations get practised. | Huang et al. 2024 (replay of pre-collision states) | `train.Worker` |
| 7 | **Learning rate 1×10⁻⁴ in stages 2 and 3** (3×10⁻⁴ in stage 1), annealed to 10%. | Smaller, safer steps when fine-tuning an already capable policy. | Quadrotor-swarm training setup: learning rate 1×10⁻⁴ | `curriculum.STAGES` |
| 8 | **Wider network: 128 features.** Trunk 384→256→128, set encoders 7→64→128 with 4 attention heads. That is about 284k actor and 284k critic parameters, up from 72k each. | More capacity for the avoidance behaviour the teacher shows. | Quadrotor-swarm training setup: hidden size 256 (we use 128 to keep laptop time reasonable) | `policy.Encoder(width)` |
| 9 | **Selection on 10 seeds per scenario** (was 5). | Less noisy choice of `best.pt`. | Henderson et al. 2018; Agarwal et al. 2021 | `curriculum.COMMON` |
| 10 | **Pipeline version 2.** First-run folders are moved to `models/<name>-old-<date>` automatically. | Old and new runs never mix, and nothing is deleted. | — | `curriculum._archive_outdated` |

What did **not** change: stage 1's recipe (it worked), PPO itself (clip 0.2, γ 0.99, λ 0.95, 5 epochs, KL stop 0.03, gradient clipping 0.5), the separate critic, reward scaling, time-limit bootstrapping, the budgets (1M / 2M / 2M PPO steps), and the held-out showcase test.

## 22. Hyperparameters: ours compared with the references

| Setting | SwarmFlow v2 | PPO paper, MuJoCo (Schulman et al. 2017) | Quadrotor-swarm training script (Batra et al. 2021 / Huang et al. 2024) |
| --- | --- | --- | --- |
| Algorithm | PPO, clipped, parameter sharing | PPO, clipped | PPO-family (Sample Factory), parameter sharing |
| Learning rate | 3×10⁻⁴ (stage 1), 1×10⁻⁴ (stages 2–3), annealed to 10% | 3×10⁻⁴ | 1×10⁻⁴ |
| Discount γ / GAE λ | 0.99 / 0.95 | 0.99 / 0.95 | 0.99 / 1.0 |
| Data per update | 16,384 drone transitions (8 envs × 256 steps × 8 drones), minibatch 2,048, 5 epochs | 2,048 steps, minibatch 64, 10 epochs | rollout 128, batch 1,024 |
| Entropy bonus | 0 | 0 | 0 |
| Actor / critic | separate networks, tanh | separate networks, tanh | separate networks, tanh |
| Hidden width | 128 | 64 | 256 |
| Agents / neighbours seen | 8 / 6 nearest | 1 / – | 8 / 6 nearest |
| Neighbour encoder | attention | – | attention |
| Collision handling | −100 to the drones involved, smooth warning within 1 m, the episode ends | – | per-collision penalty with smooth fall-off, eased in over the first 300M steps; the episode continues |
| Replay of hard moments | pre-collision restore in stages 2–3 | – | replay of pre-collision states (probability 0.75) |
| Imitation | DAgger + decaying imitation loss | – | none (pure RL) |
| Budget | 5M environment steps (40M drone transitions) | 1M steps per task | about 1 billion steps |

The quadrotor-swarm column is recalled from the public `quad-swarm-rl` training script; check the exact values there before quoting them.

**Why we don't copy everything:**

- **Budget.** 1 billion steps at our speed would take about 35 days on this laptop. Imitation from our own teacher replaces most of that budget.
- **Actions.** Their policy commands rotor thrusts on a full quadrotor model; ours commands velocities to a stabilized model, which is an easier control problem.
- **Collisions.** In their simulator a collision does not end the episode. In our benchmark it does, so instead of letting episodes run on after a crash, we replay the moments before it.

## 23. New time estimate (measured on this Mac)

With the 128-wide network and the imitation loss, rollouts run at about 1,000 environment steps/s and one PPO update takes about 4 s, so training runs at about **333 steps/s** overall (up to 4 PyTorch threads).

| Part | Approximate time |
| --- | --- |
| Stage 1: cloning + DAgger (~8 min), 1M PPO steps, validation | about 1 h |
| Stage 2: cloning + DAgger on obstacle scenarios (~15 min), 2M PPO steps, validation | about 2 h |
| Stage 3: the same on all 8 scenarios | 2–2.5 h |
| Showcase test | about 10 min |
| **Total for one seed** | **about 6 h** |

## 24. What to watch in the new logs

- **`[clone]` lines in stages 2 and 3**: `driver failures` in the DAgger rounds should be low, meaning the student already avoids objects the way the teacher does.
- **`[eval:start]` of stage 2**: crossing and head-on should now succeed from the start. The first run started at 0/5.
- **`imitation` on each PPO line**: it should stay small, close to the cloning error. A steady rise means PPO is moving away from the teacher; the validation score shows whether that helps or hurts.
- **`std` on each PPO line**: it starts around 0.20 in each stage and never goes below 0.05.
- **`failures(last 100)` in stages 2 and 3**: it should stay well below the 35–50% of the first run.
- **The validation score** in stages 2 and 3 should hold steady or rise, not fall after the first 100k steps as it did in the first run.

## 25. How to run version 2

```sh
caffeinate -i make run train      # or: make training (without the viewer)
```

- The first-run folders (`models/stage1-seed0` … `models/showcase`) are moved to `models/<name>-old-<date>` automatically when the pipeline starts, and all three stages train again. Nothing is deleted.
- Everything else is as before: results go to `models/stageN-seed0/`, the best model to `models/showcase/`, the viewer follows the newest best model, and re-running the command continues after an interruption.

**New references in this part:**

- Schmitt et al., *Kickstarting Deep Reinforcement Learning* (2018): <https://arxiv.org/abs/1803.03835>
- The quadrotor-swarm training code used by Batra et al. and Huang et al.: <https://github.com/Zhehui-Huang/quad-swarm-rl> (the repository linked in `docs/research-plan.md`)

---
---

# Part D: training version 3 (after the second full run) — current

## 26. What the second run (version 2) showed

| Stage | Result |
| --- | --- |
| **1** | Still perfect: 0.996. |
| **2** | **Fixed.** Validation success on crossing and head-on reached 100%, the best score was 0.956, and it stayed stable until the end (0.949). |
| **3** | The best score (0.861) came only 200k steps in. After that the score swung between 0.57 and 0.86, and formation precision drifted: hold slot error rose from 0.02 to 0.15–0.25 m. |
| **Showcase** | 148/160 episodes without failure (was 138) and 66/80 disturbances recovered (was 38), against 160/160 and 80/80 for the baseline. Still **NOT LEARNED YET**. All 12 failures were collisions: slalom 8, head-on 2, double crossing 2. |

## 27. Root cause, with the evidence

1. **Every test collision was a graze.** The overlap was at most 2 cm. 7 were drone–object; 5 were drone–drone, all in slalom, where two neighbours squeeze together while going around a sphere.
2. **The teacher itself passes at the edge.** The hand-coded controller keeps only **0.08 m** clearance in every obstacle scenario (minimum clearance 0.078–0.092 m). The student copies these knife-edge passes, and its velocity differs from the teacher's by 0.5–0.6 m/s near objects (0.06 m/s far from them). Any imitation error therefore turns into contact.
3. **Imitation and reward pulled in opposite directions.** The reward warns from 1 m away, while the imitation target passes at 8 cm. As the imitation weight faded from 1.0 to 0.1, stage 3 drifted: the imitation loss rose from 0.002 to 0.005, and formation precision and recovery got worse.
4. **Slalom is a different kind of problem.** Its static spheres sit on the formation slots for several seconds, so drones must leave their slots and squeeze past each other. It was the hardest scenario for every student we tried.
5. **The training log hid replayed crashes.** With crash replay, a crash followed by a successful retry was not counted as a failure. The log showed 0–6% failures in stage 3, while there were really 815 collisions in 4,049 episodes (about 20%).

## 28. The experiments that decided the fix (no full training needed)

**Teacher alone, with wider safety margins** (8 scenarios × 6 unseen seeds):

| Teacher margins (soft / hard) | Failures | Minimum clearance | Disturbances recovered | Slalom slot error |
| --- | --- | --- | --- | --- |
| 0.35 / 0.08 (baseline) | 0/48 | 0.078 m | 24/24 | 0.42 m |
| 0.5 / 0.2 | 0/48 | 0.203 m | 24/24 | 0.53 m |
| **0.6 / 0.25 (chosen)** | **0/48** | **0.260 m** | 24/24 (slowest 6.6 s) | 0.61 m |
| 0.7 / 0.3 | 0/48 | 0.308 m | 24/24 (slowest 8.4 s) | 0.72 m |

**Student after imitation only** (3 rounds of cloning + DAgger from the stage 1 model; 4 obstacle scenarios × 6 unseen seeds):

| Version | Failures | Recovered | Crossing | Head-on | Double crossing | Slalom |
| --- | --- | --- | --- | --- | --- | --- |
| A: baseline teacher (what version 2 did) | 23/24 | 2/19 | 1/6 | 0/6 | 0/6 | 0/6 |
| **B: cautious teacher 0.6 / 0.25** | **8/24** | **20/24** | **6/6** | **6/6** | 3/6 | 1/6 |
| C: cautious teacher + 5× weight on near-object states | 14/24 | 11/23 | 5/6 | 0/6 | 3/6 | 2/6 |

What the experiments show:

- The **cautious teacher** is the fix: 23 failures became 8, before any PPO.
- **Extra weight on near-object states made things worse**, so it is off by default. It is still available as `--hard-weight`.
- **Slalom stays the weakest** scenario even with the cautious teacher.

## 29. What version 3 changes

| # | Change | Why | Code |
| --- | --- | --- | --- |
| 1 | While demonstrating, the teacher keeps a **0.6 m soft / 0.25 m hard** clearance margin. The baseline you compare against keeps its own 0.35 / 0.08, unchanged. | The student learns passes with about 26 cm to spare, so small imitation errors no longer cause contact. It also agrees with the reward's 1 m warning. | `baseline.PredictiveController(soft_margin, hard_margin)`, `train(teacher_margins=...)` |
| 2 | The imitation weight goes from **1.0 to 0.5** over each stage (was 1.0 → 0.1). | The teacher never fails, so the student should stay close to it. Stage 3 degraded exactly as the weight faded. | `curriculum.COMMON` (`bc_coef_end`) |
| 3 | **Slalom is removed** from stage 3 and from the showcase. The benchmark now has 7 scenarios and 140 test episodes. | Static spheres on the formation slots are a separate problem, and the one where the learned policy failed most. It remains in the environment, in the baseline tests, and in `swarmflow eval --scenarios slalom`. | `curriculum.SHOWCASE_SCENARIOS` |
| 4 | `failures(last 100)` now counts episodes whose crash was replayed. | An honest training curve. | `train.Worker` |
| 5 | Weighting of near-object states is available but **off** (`--hard-weight 1`). | It hurt in experiment C. | `train._demo_tensors` |
| 6 | Pipeline version 3: version-2 folders are moved to `models/<name>-old-<date>` automatically. | Old and new runs never mix. | `curriculum.PIPELINE_VERSION` |

Everything else from Part C stays: cloning + DAgger in every stage, the imitation loss inside PPO, the exploration floor, per-drone crash penalties, the 1 m warning, crash replay in stages 2–3, learning rate 1×10⁻⁴ in stages 2–3, and the 128-wide network.

A tiny end-to-end run (stage 1 → 2 → 3 → showcase → viewer report, 64 steps per stage) and all 38 unit tests pass with these settings. So a long run will not stop on a code error, but only the full run can show the final quality.

## 30. What to expect and watch

- **`[clone]` in stages 2 and 3**: fewer `driver failures` in the DAgger rounds than version 2 had (15/52 in stage 2, 11/47 in stage 3).
- **`[eval:start]` of stage 2**: crossing and head-on should succeed immediately. Version 2 started at 1/10 and 6/10.
- **`failures(last 100)`**: now honest, so it may look higher than version 2's numbers even when the policy is better.
- **Showcase minimum clearance**: it should be well above 0 (version 2 had −0.02 m, meaning contact).
- **Time**: about 5.5–6 h. Stage 3 has one scenario fewer.

## 31. Remaining risks, stated honestly

- The acceptance check allows **zero** failures in 140 test episodes. A learned policy can still have a rare failure. If that happens, the next steps are:
  - a larger teacher margin (`--teacher-margins 0.7 0.3`);
  - more DAgger rounds;
  - a longer stage 3;
  - more training seeds.
- The cautious teacher makes slightly bigger detours, and its slowest recovery was 6.6 s. The "95% recovered within 5 s" check can therefore still be tight on double crossings.
