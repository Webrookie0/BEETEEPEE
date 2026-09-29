# Paper 23: Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Sanmit Narvekar et al.. **Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey**. JMLR 2020.

[Paper / official source](https://arxiv.org/abs/2003.04960) · [Full text](https://arxiv.org/html/2003.04960)

**Relevant source sections:** §3 curriculum framework; §4.1 task generation; §4.2 sequencing; §4.3 knowledge transfer.

**Project status:** Framework for describing task generation, sequencing and transfer; no automatic curriculum algorithm implemented.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Tasks sequencing and transfer

The survey distinguishes what tasks are generated, their order and what knowledge transfers. Here these are seeded benchmark episodes, the three [hand-authored stages](paper_22.md#progressive-task-distribution), and a checkpoint containing policy/critic weights, optimizer state and reward-scale moments.

Conceptually stage $k+1$ initializes from the checkpoint selected in stage $k$:

$$\theta_{k+1}^{init}\leftarrow\theta_k^{best},\qquad
best=\arg\max_{checkpoint}\;score_{selection}.$$

This project equation describes existing orchestration rather than a survey algorithm. Within each stage, teacher-labelled data and PPO rollouts are recollected. Saved step counts can start from the preceding **best** step, which may be earlier than its final budget; they need not equal total compute spent across all stages.

No automatic task generation, teacher-agent curriculum, difficulty learner or transfer-learning baseline is implemented. The survey also motivates evaluating learning speed and final task quality separately. Code: [`curriculum.stage_arguments`](../src/swarmflow/curriculum.py#L62), [`train.train`](../src/swarmflow/train.py#L228), [`curriculum.training_chain`](../src/swarmflow/curriculum.py#L159).

## Repository evidence

[training §6](../training.md), [curriculum.py](../src/swarmflow/curriculum.py).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
