# Paper 13: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Stéphane Ross, Geoffrey J. Gordon and J. Andrew Bagnell. **A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning**. AISTATS 2011.

[Paper / official source](https://arxiv.org/abs/1011.0686) · [Full text](https://arxiv.org/html/1011.0686)

**Relevant source sections:** §2 supervised imitation and distribution shift; §3 Dataset Aggregation; §4 guarantees and assumptions.

**Project status:** Implemented student-driven dataset aggregation after a noisy-teacher initial round.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Dataset aggregation

Let $\pi_T$ be the predictive teacher and $\pi_k$ the current student. After the initial teacher round, the student flies while the teacher labels each state it visits:

$$D_{k+1}=D_k\cup\{(o,\pi_T(o)):o\sim d_{\pi_k}\},\qquad
\theta_{k+1}\approx\arg\min_\theta L_{BC}(D_{k+1}).$$

The dataset retains all earlier rounds within the stage. Three extra DAgger rounds mean **four rounds total**, not three. Each round collects `clone_steps` team steps and contributes one example per drone per step. Stage 1: $4\times10{,}000\times8=320{,}000$ labels. Stages 2 and 3: 640,000 labels each at default counts. New stage cloning uses that stage's scenarios and a fresh stage dataset. On continuation, data may be rebuilt from current student visits rather than restored byte-for-byte.

## Cloning loss and fitting

$$L_{BC}=\frac{1}{3|D|}\sum_{(o,a_T)\in D}\|\tanh\mu_\theta(o)-a_T\|_2^2.$$

This is PyTorch mean squared error over examples and action coordinates. Labels are normalized desired-velocity actions. Fitting uses actor parameters only, Adam learning rate $5\times10^{-4}$, minibatches of 1,024, 15 epochs by default and gradient norm clipping 0.5. The teacher sees the same local observation schema as the student.

## Optional hard-state weighting

A label is "hard" when any valid obstacle has surface gap <2 m or any valid neighbor has gap <0.6 m. If `hard_weight=w`, its sampling weight is

$$w_k=1+(w-1)\mathbf1_{hard,k},\qquad P(k)=w_k/\sum_jw_j.$$

The code samples using a cumulative distribution rather than multiplying the MSE directly. The default is 1 (uniform); larger weights were tested and rejected for the default recipe. This prioritization is a local extension, not DAgger's standard algorithm.

Code: [`train.clone`](../src/swarmflow/train.py#L184), [`train._collect`](../src/swarmflow/train.py#L155), [`train._fit_clone`](../src/swarmflow/train.py#L137), [`train._demo_tensors`](../src/swarmflow/train.py#L117), [`train._demo_ids`](../src/swarmflow/train.py#L129).

## Differences and scope

The source allows teacher/student mixtures during rollout; after round zero this code flies the deterministic student with no mixture. Neural-network fitting is approximate and finite-data. DAgger's theoretical improvements under its assumptions do not guarantee zero drone collisions. The initial noisy round follows the [DART-inspired approach](paper_14.md#noisy-teacher-clean-labels).

## Repository evidence

[training §7 and Parts C–E](../training.md), [curriculum stages](../src/swarmflow/curriculum.py), and clone/resume tests in [test_learning.py](../tests/test_learning.py).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
