# Paper 27: Kickstarting Deep Reinforcement Learning

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Simon Schmitt et al.. **Kickstarting Deep Reinforcement Learning**. 2018.

[Paper / official source](https://arxiv.org/abs/1803.03835) · [Full text](https://arxiv.org/html/1803.03835)

**Relevant source sections:** §2.1 knowledge transfer; §2.2 kickstarting actor–critic; §2.3 population-based training; §5.2.2 distillation weights.

**Project status:** Teacher regularization with a changing weight; adapted loss and schedule.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Teacher loss during reinforcement learning

The source adds teacher/student cross-entropy to RL:

$$L_{kick}=L_{RL}+\lambda_k H(\pi_T\Vert\pi_S),\qquad
H(\pi_T\Vert\pi_S)=-\mathbb E_{a\sim\pi_T}\log\pi_S(a\mid o).$$

SwarmFlow's teacher returns a deterministic velocity action, not a full distribution. The code therefore adds [action MSE](paper_13.md#cloning-loss-and-fitting) on the stage's stored teacher-labelled data:

$$L=L_{PPO}+(1-0.5f)L_{BC}.$$

The weight decreases linearly from 1.0 to 0.5, retaining teacher guidance at the end. Older v2 runs used an end coefficient of 0.1. Teacher data are sampled separately from fresh PPO trajectories and refreshed when cloning a new stage. No online teacher-distribution cross-entropy, multi-teacher distillation, IMPALA or population-based tuning is implemented.

Code: [`train.train`](../src/swarmflow/train.py#L228) (`bc_now`, `bc_loss`). This shares motivation with [DAPG](paper_15.md#demonstrations-before-and-during-rl); the same loss should not be counted as two separate algorithms.

## Repository evidence

[training Part C and references](../training.md), [curriculum.COMMON](../src/swarmflow/curriculum.py).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
