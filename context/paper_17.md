# Paper 17: AWAC: Accelerating Online Reinforcement Learning with Offline Datasets

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Ashvin Nair, Abhishek Gupta, Murtaza Dalal and Sergey Levine. **AWAC: Accelerating Online Reinforcement Learning with Offline Datasets**. 2020 preprint.

[Paper / official source](https://arxiv.org/abs/2006.09359) · [Full text](https://arxiv.org/html/2006.09359)

**Relevant source sections:** §III challenges in offline RL with online fine-tuning; §IV advantage-weighted actor critic.

**Project status:** Background for fine-tuning from demonstrations; AWAC algorithm not implemented.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Fine-tuning context

AWAC studies learning from offline datasets and improving online, including bootstrap-error and policy-update difficulties. The repository uses it as motivation for preparing a critic and preserving imitation skills before and during PPO. The exact [warm-up](paper_16.md#critic-warm-up-motivation) and [imitation loss](paper_15.md#demonstrations-before-and-during-rl) are project adaptations.

For comparison only, the AWAC actor objective weights data log likelihood by an exponential advantage, schematically

$$\max_\theta\;\mathbb E_{(s,a)\sim D}\left[\exp(A(s,a)/\lambda)\log\pi_\theta(a\mid s)\right].$$

That equation is **not executed in SwarmFlow**. There is no off-policy Q critic, AWAC replay buffer, exponential-advantage demonstration weighting or AWAC update. Teacher examples affect MSE imitation; PPO advantages come from fresh on-policy rollouts. This distinction also applies to collision-state resets, which are not off-policy transition replay.

## Repository evidence

[training §7, §11.14 and reading list](../training.md), [train.py](../src/swarmflow/train.py).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
