# Paper 15: Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Aravind Rajeswaran et al.. **Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations**. RSS 2018.

[Paper / official source](https://arxiv.org/abs/1709.10087) · [Full text](https://arxiv.org/html/1709.10087)

**Relevant source sections:** §IV-C1 pretraining with behavior cloning; §IV-C2 RL fine-tuning with augmented loss, including equation (6).

**Project status:** Imitation warm start and demonstration regularization adapted to PPO, not the full DAPG optimizer.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Demonstrations before and during RL

The source's augmented policy gradient contains on-policy advantage terms and weighted demonstration log-likelihood terms:

$$g_{aug}=\sum_{(s,a)\in D_\pi}\nabla_\theta\log\pi_\theta(a\mid s)A^\pi(s,a)
+\sum_{(s,a)\in D_T}\nabla_\theta\log\pi_\theta(a\mid s)w(s,a).$$

SwarmFlow adopts the strategy: initialize the actor from demonstrations, then keep a demonstration term while optimizing RL. Its implementation instead minimizes squashed deterministic-action MSE with PPO:

$$L=L_{PPO}+\beta(f)L_{BC},\qquad\beta(f)=1-0.5f.$$

$f$ is progress through the current stage's PPO budget. A separate demonstration minibatch of 1,024 is sampled for each PPO minibatch. The dataset stays fixed during the stage's PPO portion; it was aggregated during cloning. Warm-up updates train only the critic. See [PPO total loss](paper_07.md#combined-loss) and [cloning formula](paper_13.md#cloning-loss-and-fitting).

## Differences and scope

DAPG's natural-gradient update and demonstration log-likelihood weighting are not reproduced. SwarmFlow uses Adam, PPO clipping, MSE to its own predictive teacher, and a linear coefficient schedule that stays at least 0.5 in the default curriculum. This makes "PPO with a decaying imitation term" the precise description. The paper's dexterous-hand tasks and measured gains are unrelated evidence, not SwarmFlow test results.

Code: [`train.train`](../src/swarmflow/train.py#L228) (`demo`, `bc_now`, `bc_loss`), [`train.clone`](../src/swarmflow/train.py#L184).

## Repository evidence

[training Parts C and D](../training.md), [knowledge §5](../knowledge.md#5-research-papers-what-we-took-from-each), [continued-run imitation tests](../tests/test_learning.py).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
