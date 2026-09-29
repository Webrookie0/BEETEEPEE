# Paper 12: Time Limits in Reinforcement Learning

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Fabio Pardo, Arash Tavakoli, Vitaly Levdik and Petar Kormushev. **Time Limits in Reinforcement Learning**. ICML 2018.

[Paper / official source](https://arxiv.org/abs/1712.00378) · [Full text](https://arxiv.org/html/1712.00378)

**Relevant source sections:** §2 time-awareness for time-limited tasks; §3 partial-episode bootstrapping for time-unlimited tasks.

**Project status:** Implemented distinction between failure termination and artificial time-limit truncation.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Termination and truncation

The default 24 s collection limit is treated as truncation. Collision, terrain contact and world-boundary failure terminate the whole team. A failure takes precedence over a simultaneous time limit. Continuous viewer mode suppresses time truncation.

$$d_t=\mathbf1_{failure},\qquad
e_t=\mathbf1_{failure\lor time\ limit},\qquad
y_t=\tilde r_t+\gamma(1-d_t)V(o_{t+1}^{final}).$$

GAE continuation is multiplied by $1-e_t$. Thus truncation permits bootstrapping from the final observation but prohibits carrying advantages into the reset episode; termination permits neither. A restored collision state is still a new rollout segment after a truly terminal failed transition.

Code: [`env.SwarmEnv.step`](../src/swarmflow/env.py#L307), [`train.Worker.step`](../src/swarmflow/train.py#L80), [`train.advantages`](../src/swarmflow/train.py#L26), [`train.train`](../src/swarmflow/train.py#L228).

## Remaining time and interpretation

The observation also contains $\max(0,1-t/T)$ as its final self feature. The source separates intrinsically finite-horizon objectives from artificially truncated continuing tasks. This implementation follows continuing-task bootstrapping despite exposing time remaining; the presence of the clock alone does not mean the 24 s limit is a terminal failure. Reinterpreting the task as a true finite-horizon objective would require revisiting this choice.

Tests: `test_time_limit_is_truncation` and `test_gae_bootstraps_truncation_without_crossing_reset`. Correct final-observation bootstrap is important because workers reset immediately after episode end.

## Repository evidence

[research plan: environment contract](../docs/research-plan.md), [test_environment.py](../tests/test_environment.py), [test_learning.py](../tests/test_learning.py).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
