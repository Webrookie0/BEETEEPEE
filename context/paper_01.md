# Paper 01: Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Sumeet Batra, Zhehui Huang, Aleksei Petrenko, Tushar Kumar, Artem Molchanov and Gaurav S. Sukhatme. **Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning**. CoRL 2021.

[Paper / official source](https://arxiv.org/abs/2109.07735) · [Full text](https://arxiv.org/html/2109.07735)

**Relevant source sections:** §3.1 problem formulation; §3.3 training setup; §§3.4–3.5 architectures and neighborhood encoder; §4.4 obstacle avoidance; supplement I and II.

**Project status:** Implemented with substantial task, dynamics, architecture and reward adaptations.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Shared local policy and relative observations

Every drone runs the same actor parameters, using its own observation:

$$a_{i,t}\sim\pi_\theta(\cdot\mid o_{i,t}),\qquad
\Delta p_{ij}=p_j-p_i,\quad\Delta v_{ij}=v_j-v_i.$$

The paper motivates local neighbor position/velocity and a permutation-invariant neighborhood representation. SwarmFlow uses the 17-component self vector and masked 8-component neighbor/object rows described in [observations](project_context.md#observations). These include the assigned slot, target velocity, previous command, radius and visibility. The actor receives no future trajectories, event times, global centroid or other drones' goals. The simulator can use privileged state for reward and collision measurement; the policy cannot.

Code: [`env.SwarmEnv._observe`](../src/swarmflow/env.py#L240), [`policy.ActorCritic`](../src/swarmflow/policy.py#L86), [`policy.SetEncoder`](../src/swarmflow/policy.py#L30). The precise pooling equations are in [Paper 19](paper_19.md#attention-pooling).

## Persistent formation and moving obstacles

The paper's §4.4 already demonstrates breaking and re-establishing formations around moving spherical obstacles. This motivates the benchmark; the behavior itself is not a new claim.

SwarmFlow's own persistent world-aligned slot rule is

$$s_i(t)=q(t)-(d_{stand},0,0)+\delta_i,\qquad e_i(t)=\|p_i(t)-s_i(t)\|_2.$$

Each drone keeps its offset $\delta_i$ for the episode. The target is exogenous. Slots do not rotate instantly with target heading. Terrain worlds add a project-specific altitude adjustment. Crossing objects move at constant velocity and are measured by per-event recovery; see [scenario formulas](project_context.md#scenarios-and-worlds) and [recovery](project_context.md#recovery).

Code: [`env.SwarmEnv.slots`](../src/swarmflow/env.py#L64), [`scenarios.crossing`](../src/swarmflow/scenarios.py#L156), [`env.SwarmEnv._update_recovery`](../src/swarmflow/env.py#L280).

## Reward components and collision attribution

The source separates position, collision and auxiliary terms. Its §3.3 collision expression uses a collision indicator and a smooth proximity penalty; rewritten with generic weights:

$$r_{col,i}=-\alpha_{col}\mathbf1_{collision,i}
-\alpha_{prox}\sum_j\max(1-\|\Delta p_{ij}\|/d_{prox},0).$$

The supplement scales continuous costs by the control interval, while a collision penalty is an event cost. SwarmFlow adopts those principles but uses surface clearances, quadratic warning costs, a capped tracking cost and terminal failures:

$$r_i=\Delta t(1-C_i)-100\mathbf1_{failure}\mathbf1_{culprit,i}.$$

Only involved drones receive the 100-point failure penalty; every drone's episode ends. The obstacle warning begins at a 1 m surface gap. This is **not the paper's exact reward**: the complete code formula and its bounds are in [project reward](project_context.md#reward).

Code: [`env.SwarmEnv.step`](../src/swarmflow/env.py#L307), [`env.SwarmEnv._reward`](../src/swarmflow/env.py#L392). Tests: `test_crash_penalty_goes_to_the_drones_involved`, `test_obstacle_warning_starts_one_metre_away` in [test_learning.py](../tests/test_learning.py).

## Differences and scope

The paper commands rotor thrusts and models attitude/angular velocity, collisions and physical deployment. SwarmFlow commands desired velocity in a stabilized reduced-order model. Its attention pooling differs from the paper's neighbor attention, and its PPO loop is synchronous batched inference with sequential environment stepping rather than the paper's high-throughput asynchronous Sample Factory setup. No source code was imported from QuadSwarm. Paper training budgets and flight results are not SwarmFlow results.

## Repository evidence

[Research plan](../docs/research-plan.md), [sofar](../sofar.md), [knowledge §5](../knowledge.md#5-research-papers-what-we-took-from-each), and [training history](../training.md), especially the v2 per-drone reward changes.

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
