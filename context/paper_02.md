# Paper 02: Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep Reinforcement Learning

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Zhehui Huang, Zhaojing Yang, Rahul Krupani, Baskın Şenbaşlar, Sumeet Batra and Gaurav S. Sukhatme. **Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep Reinforcement Learning**. ICRA 2024.

[Paper / official source](https://arxiv.org/abs/2309.13285) · [Full text](https://arxiv.org/html/2309.13285)

**Relevant source sections:** §III-B training setup and observations; §III-C model architecture; §III-D replay buffer; §IV-A ablations; §V limitations.

**Project status:** Adapted obstacle representation and collision-state replay; optional rays/gap features; dense full-height benchmark not reproduced.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Bounded obstacle sensing

The source motivates distance-aware obstacle inputs and attention. Its main environment has static obstacles extending through the room height. SwarmFlow instead selects up to six nearest visible spheres within 7 m, pads to a fixed tensor, and masks missing rows. Unobserved objects still participate in collision checks.

$$g_{ij}=\|p_j-p_i\|_2-r_i-r_j.$$

Rows contain relative xyz, relative velocity xyz, radius and validity. Noise is $\epsilon\sim\mathcal N(0,\sigma_{obs}^2I)$ on relative position; independent dropout clears real-object/neighbor rows. Neither is a complete sensor model. They are project options, not a reproduction of this paper's sensing stack. Defaults are ideal observations: noise = dropout = 0. Optional planar ray hits fill only empty obstacle rows, using [ray/sphere intersections](project_context.md#ray-intersections).

Code: [`env.SwarmEnv._object_rows`](../src/swarmflow/env.py#L189), [`env.SwarmEnv._fill_ray_rows`](../src/swarmflow/env.py#L220). Attention is independently pooled over two sets, unlike the paper's fusion of neighbor and obstacle embeddings.

## Pre-collision state replay

The source's §III-D stores states 1.5 s before collisions and samples replay starts with probability $\alpha_r$; a state is removed after a replay-count limit. SwarmFlow adapts the idea without copying that sampling scheme:

1. A `Worker` records successful environment states every `max(1, int(2/dt))` steps, 40 steps at defaults.
2. A deque retains at most four snapshots.
3. On a collision with a nonempty deque, restore **the oldest retained snapshot**, `ring[0]`, at most twice during that episode.
4. Clear the deque, collect fresh current-policy transitions, and retain the collision count. A successful retry does not erase the failed attempt in the recent episode statistic.

At default settings, the oldest snapshot can be roughly 0–8 seconds old depending on episode timing and how full the deque is. Earlier prose saying "exactly/about 2 seconds before collision" is not the actual selection rule. World-boundary failures without `collision=True` do not trigger this branch. These resets are enabled in curriculum stages 2–3.

$$S\leftarrow S_{oldest},\qquad a_t\sim\pi_{\theta_{current}}(\cdot\mid o(S_t)).$$

This is replay of a **starting state**, not reuse of stale PPO actions, returns or likelihoods. Terminal masks and reset masks still cut the old trajectory; see [Paper 12](paper_12.md#termination-and-truncation). Capture/restore copies simulation arrays, events, objects, terrain peaks and observations, but not RNG state; it is not exact stochastic replay.

Code: [`train.Worker.step`](../src/swarmflow/train.py#L80), [`env.SwarmEnv.capture_state`](../src/swarmflow/env.py#L436), [`env.SwarmEnv.restore_state`](../src/swarmflow/env.py#L448).

## Optional gap features

With `--gap-features`, each row adds three project-designed features:

$$\left[g/2,\;\exp(-\max(g,0)/0.3),\;c/3\right],\qquad
c=-\frac{\Delta p\cdot\Delta v}{\max(\|\Delta p\|,10^{-6})}.$$

The radius used for the observing drone in this feature is `DRONE_RADIUS_M = Config().radius`, i.e. 0.18 m; it is not taken from an arbitrary changed runtime configuration. The option is saved in checkpoint network metadata. These features were tested and are **off** in pipeline v4 after worse development results. They are inspired by distance observations, not the source's complete signed-distance representation.

Code: [`policy.SetEncoder.forward`](../src/swarmflow/policy.py#L46). Tests cover permutation invariance and checkpoint round-trip with the option enabled in [test_learning.py](../tests/test_learning.py).

## Repository evidence

[Research plan](../docs/research-plan.md), [training Parts C and E](../training.md), [knowledge §5](../knowledge.md#5-research-papers-what-we-took-from-each).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
