# Paper 30: Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Maciej Wołczyk et al.. **Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem**. ICML 2024.

[Paper / official source](https://arxiv.org/abs/2402.02868) · [Full text](https://arxiv.org/html/2402.02868)

**Relevant source sections:** §2 forgetting of pretrained capabilities; §§4–5 retention results and analysis; appendix C retention methods.

**Project status:** Diagnosis and retention motivation; no separate paper-specific fine-tuning algorithm.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Retaining pretrained behavior

The source analyzes loss of pretrained capabilities during RL fine-tuning. The project invokes it when interpreting archived run 3: stage 2 and 3 best checkpoints were their post-cloning starting evaluations, and later PPO did not improve their selection scores.

The implementation responses are retaining [teacher MSE during PPO](paper_15.md#demonstrations-before-and-during-rl), practicing earlier tasks, validating the cloned starting point, preserving the best checkpoint and reducing stages 2–3 to 1 M PPO steps each in v4.

$$checkpoint_{best}=\arg\max_{c\in\{postclone,periodic,final\}}score_{selection}(c).$$

This is the project's selection rule, not an equation claimed from the source. More PPO steps may improve or degrade behavior, so `latest.pt` and `best.pt` serve different purposes. A model selected at the cloned start does not establish additional improvement from PPO in that stage.

## Limits of the diagnosis

The code does not compute representational forgetting, CKA or the paper's experimental retention benchmarks. Validation degradation alone does not isolate forgetting from exploration, stochastic optimization, reward mismatch or changed data coverage. No paper-specific numerical result is claimed for SwarmFlow.

Code: [`train.train`](../src/swarmflow/train.py#L228) (`validate`, `best_score`, imitation coefficient), [`curriculum.stage_arguments`](../src/swarmflow/curriculum.py#L62). Historical evidence: [saved runs](history.md#saved-training-runs).

## Repository evidence

[training Parts D and E](../training.md), [previous-runs README](../models/previous-runs/README.md), archived `evaluations.jsonl` and `best.json` files.

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
