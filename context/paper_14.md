# Paper 14: DART: Noise Injection for Robust Imitation Learning

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Michael Laskey et al.. **DART: Noise Injection for Robust Imitation Learning**. CoRL 2017.

[Paper / official source](https://arxiv.org/abs/1703.09327) · [Full text](https://arxiv.org/html/1703.09327)

**Relevant source sections:** §4 off-policy imitation with noise injection; §4.1 DART; §4.2 theoretical analysis.

**Project status:** Fixed-noise teacher data collection inspired by DART; the noise optimization procedure is not implemented.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Noisy teacher clean labels

In round zero, the teacher computes a clean label but the environment executes a noisy command:

$$a_T=\pi_T(o),\qquad a_{exec}=\operatorname{clip}(a_T+\epsilon,-1,1),\qquad
\epsilon\sim\mathcal N(0,\sigma_{clone}^2I).$$

The saved example is $(o,a_T)$, not $(o,a_{exec})$. The next observation reflects the noisy execution, so the dataset contains off-nominal states paired with corrective teacher actions. Default `clone_noise=0.3` is in normalized action-component units; component clipping is followed by the environment's velocity norm limit. Sensor noise is a separate configuration option.

## Adaptation and use

The paper optimizes injected noise to approximate learner errors. SwarmFlow uses a fixed scalar standard deviation and does not solve that noise-selection objective, fit a covariance, or reproduce the paper's guarantees. Subsequent rounds use [DAgger](paper_13.md#dataset-aggregation), not continued noisy-teacher flight. Code: [`train._collect`](../src/swarmflow/train.py#L155), [`train.clone`](../src/swarmflow/train.py#L184).

## Repository evidence

[training §7 and §11.13](../training.md), [train.py](../src/swarmflow/train.py).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
