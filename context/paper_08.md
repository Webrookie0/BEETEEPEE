# Paper 08: High-Dimensional Continuous Control Using Generalized Advantage Estimation

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

John Schulman et al.. **High-Dimensional Continuous Control Using Generalized Advantage Estimation**. ICLR 2016; preprint 2015.

[Paper / official source](https://arxiv.org/abs/1506.02438) · [Full text](https://arxiv.org/html/1506.02438)

**Relevant source sections:** §3 advantage function estimation; §5 value estimation.

**Project status:** Implemented GAE with separate failure-bootstrap and reset-continuation masks.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## GAE recurrence

The source estimator is an exponentially weighted sum of TD residuals. The exact finite-rollout implementation uses:

$$\delta_t=\tilde r_t+\gamma(1-d_t)V(o^{final}_{t+1})-V(o_t),$$
$$\hat A_t=\delta_t+\gamma\lambda(1-e_t)\hat A_{t+1},\qquad
\hat R_t=\hat A_t+V(o_t).$$

$d_t$ marks true termination, $e_t$ marks termination **or** truncation, and $\tilde r_t$ is scaled reward when scaling is enabled. The backward accumulator starts at zero beyond the rollout. Within a continuous unbroken trajectory, this is $\hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}$. Default $\gamma=0.99$, $\lambda=0.95$.

At a normal rollout boundary, the final critic value bootstraps the last transition. At episode end, the trainer explicitly evaluates the terminal observation before the `Worker` starts/restores another episode. The reset observation is never substituted for that transition's final value. See [time limits](paper_12.md#termination-and-truncation).

## Scale and use

The effective discount scale $1/(1-\gamma)=100$ control steps is about 5 s at 20 Hz; it is a rule of thumb, not a strict cutoff. Advantages are then batch-standardized for PPO, while return targets retain the critic's scaled-reward units. Increasing $\lambda$ generally uses longer residual sums and changes the bias/variance tradeoff; this project does not conduct a GAE parameter sweep.

Code: [`train.advantages`](../src/swarmflow/train.py#L26), [`train.train`](../src/swarmflow/train.py#L228). The exact reset/truncation numerical case is checked by `test_gae_bootstraps_truncation_without_crossing_reset` in [test_learning.py](../tests/test_learning.py).

## Repository evidence

[training](../training.md), [train.advantages](../src/swarmflow/train.py), and the GAE unit test.

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
