# Paper 25: Deep Reinforcement Learning that Matters

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Peter Henderson et al.. **Deep Reinforcement Learning that Matters**. AAAI 2018; preprint 2017.

[Paper / official source](https://arxiv.org/abs/1709.06560) · [Full text](https://arxiv.org/html/1709.06560)

**Relevant source sections:** Experimental Analysis: hyperparameters, random seeds/trials and codebases; Reporting Evaluation Metrics.

**Project status:** Reproducibility, multiple seeds and explicit settings; not a performance guarantee.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Multiple seeds and reproducibility

`make train-full-multi` requests independent training seeds 0, 1 and 2. Logs preserve configuration, stage/network versions, scenario seeds, raw losses, failures and selection evaluations. Multiple episode seeds from one trained network are not equivalent to multiple training seeds.

For episode metric $x_j$, the code records

$$\bar x=\frac1n\sum_jx_j,\qquad s=\sqrt{\frac{\sum_j(x_j-\bar x)^2}{n-1}}.$$

The score-based showcase picks the best candidate; that choice should not replace reporting the other training seeds or variability. Historical archived full pipelines are seed 0 repetitions/versions, not three independent seeds of one fixed protocol.

## Logged quantities and limits

Reports retain complete episode records, failures, clearance and conditional recovery times. Checkpoints include environment and algorithm settings, but resume restores fresh episodes rather than exact random-generator/environment state. GPU/device/library differences can change reproducibility. The code writes atomic checkpoint replacements and archives older stage/showcase folders; it does not record every external hardware/software detail automatically.

Code: [`train.train`](../src/swarmflow/train.py#L228), [`evaluate.aggregate`](../src/swarmflow/evaluate.py#L37), [`curriculum.build_showcase`](../src/swarmflow/curriculum.py#L178). Exact statistical definitions: [metrics](project_context.md#metrics-and-selection).

## Repository evidence

[Makefile](../Makefile), [training §7 and reading list](../training.md), [historical run records](history.md#saved-training-runs).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
