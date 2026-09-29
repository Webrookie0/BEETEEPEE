# Paper 16: Jump-Start Reinforcement Learning

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Ikechukwu Uchendu et al.. **Jump-Start Reinforcement Learning**. ICML 2023; preprint 2022.

[Paper / official source](https://arxiv.org/abs/2204.02372) · [Full text](https://arxiv.org/html/2204.02372)

**Relevant source sections:** §4.1 rolling in with two policies; §4.2 algorithm; §5.1 imitation-plus-RL comparisons; appendix A.1.

**Project status:** Fine-tuning motivation; JSRL guide-policy roll-in algorithm is not implemented.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Critic warm-up motivation

The repository cites this work when discussing how an imitation-initialized policy can lose its skills during RL fine-tuning. JSRL itself uses guide-policy roll-ins and an exploration-policy curriculum. Those mechanics are absent here.

SwarmFlow's engineering response is to train the value function first:

$$L_{warm}=0.5L_V=0.25\operatorname{mean}(V-\hat R)^2,\qquad\nabla_\theta L_{warm}\leftarrow0.$$

With cloning, the default is ten critic-only PPO updates; without cloning, resuming from another folder defaults to five, otherwise zero. Since **all current stages clone**, their normal default is ten, including stages 2–3. Warm-up still collects and counts fresh environment steps within the PPO budget.

## Attribution boundary

The warm-up count and separate value network are local design choices, not a verbatim JSRL recipe or demonstrated optimum. This source contributes conceptual context, not a distinct JSRL loss or theorem used in the code. See also [AWAC](paper_17.md#fine-tuning-context) and [forgetting](paper_30.md#retaining-pretrained-behavior).

Code: [`train.train`](../src/swarmflow/train.py#L228) (`critic_warmup`, `warmup`, actor gradient clearing).

## Repository evidence

[training §7 and §11.14](../training.md), [train.py](../src/swarmflow/train.py).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
