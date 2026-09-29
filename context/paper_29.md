# Paper 29: Information Theoretic MPC for Model-Based Reinforcement Learning

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Grady Williams et al.. **Information Theoretic MPC for Model-Based Reinforcement Learning**. ICRA 2017.

[Paper / official source](https://ieeexplore.ieee.org/document/7989202)

**Relevant source sections:** Information-theoretic/path-integral control weighting of sampled trajectories. Exact publisher section numbering was not verified.

**Project status:** Optional MPPI-style cost-weighted candidate blend; tested and disabled in the current recipe.

**Verification:** The source link is retained from the project bibliography; publisher full text was not available during this documentation pass. Equations below describe the project code unless explicitly marked otherwise. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Temperature-weighted commands

When `PredictiveController(temperature=T)` has $T>0$, candidate costs $C_k$ become exponential weights:

$$w_k=\exp(-(C_k-\min_jC_j)/T),\qquad
v^{cmd}=\operatorname{limit}_{v_{max}}\left(\frac{\sum_kw_kv_k}{\sum_kw_k}\right).$$

Subtracting the minimum cost improves numerical conditioning without changing normalized weights. With `temperature=0`, the implementation uses the minimum-cost candidate directly, rather than dividing by zero or approximating a small temperature.

## Experimental status and differences

The history tested temperatures 0.3 and 1.0; the averaged commands slowed double-crossing recovery, so the default remains zero. The option exists on the Python controller constructor but is not a CLI training temperature flag, and `clone()` constructs its teacher without enabling it.

This is an MPPI-inspired blend over a deterministic set of constant-velocity candidates. It does not implement stochastic control-sequence rollouts, importance-sampling corrections, path-integral dynamics or the full MPPI update. Averaging candidate velocities need not preserve the safety of either candidate.

Code: [`baseline.PredictiveController.action`](../src/swarmflow/baseline.py#L19). Related default planner: [project context](project_context.md#predictive-controller).

## Repository evidence

[training §34 and Part E references](../training.md), [baseline.py](../src/swarmflow/baseline.py).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
