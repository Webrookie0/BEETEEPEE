# Paper 20: Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Ryan Lowe et al.. **Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments**. NeurIPS 2017.

[Paper / official source](https://arxiv.org/abs/1706.02275) · [Full text](https://arxiv.org/html/1706.02275)

**Relevant source sections:** §4.1 multi-agent actor critic; appendix MADDPG algorithm.

**Project status:** Centralized-training/decentralized-execution concept only; MADDPG is not implemented.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Centralized training decentralized execution

The source uses a critic with access to joint information during learning while actors remain local at execution. SwarmFlow optionally adopts that information separation:

$$a_i=\tanh\mu_\theta(o_i),\qquad
V_i=f_\phi\left(\frac1N\sum_jE_\phi(o_j)\right).$$

This is a state-value critic over pooled local observations, not MADDPG's action-conditioned $Q_i(x,a_1,\ldots,a_N)$. Its output is identical for all drones of a team. There are no deterministic policy-gradient updates, target actor/critic networks or MADDPG replay buffer. The default remains local IPPO. Full details are in [Paper 4](paper_04.md#optional-centralized-value-function).

Code: [`policy.ActorCritic.value_central`](../src/swarmflow/policy.py#L116), [`train.train`](../src/swarmflow/train.py#L228). Team grouping is preserved when shuffling central-critic minibatches.

## Repository evidence

[training §11.11 and reading list](../training.md), [central-value shape test](../tests/test_learning.py).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
