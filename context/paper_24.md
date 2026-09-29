# Paper 24: Leveraging Procedural Generation to Benchmark Reinforcement Learning

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Karl Cobbe et al.. **Leveraging Procedural Generation to Benchmark Reinforcement Learning**. ICML 2020.

[Paper / official source](https://arxiv.org/abs/1912.01588) · [Full text](https://arxiv.org/html/1912.01588)

**Relevant source sections:** §2.2 experimental protocols; §2.3 hyperparameter selection; §3 generalization experiments.

**Project status:** Separate training, selection and test generation; Procgen environments not used.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Seed separation

SwarmFlow distinguishes network/training seeds from episode-generation seeds:

1. Training and imitation episodes draw integers in **0–999**.
2. Default development evaluation starts at **1000**; historical scaling also used 2000–2001.
3. Best-checkpoint selection uses **5000–5009** in curriculum runs (ten seeds per scenario).
4. Default showcase test uses **100000–100019** (20 seeds per scenario), after selecting a model using selection scores.

Seeded reset determines initial jitter and scenario variables. This tests generalization across samples of the implemented generator, not all trajectories, worlds or sensing conditions. Avoid expanding development ranges into selection seeds. Changing geometry, density or physics is a different generalization test.

## Selection and testing

$$c^*=\arg\max_c score(D_{selection};c),\qquad
report=Eval(c^*,D_{test}).$$

`build_showcase` chooses candidates from `best.json` before evaluating learned and baseline policies with the same scenario/seed set. Test results do not enter the selection function. However, repeated human tuning after viewing those tests can still consume their held-out status; numeric seed ranges alone cannot prevent leakage. The project history already contains tests from those ranges, so reserve a fresh test set for future independent claims.

No Procgen game, generator or source-specific loss is implemented. Code: [`train.Worker.start_episode`](../src/swarmflow/train.py#L74), [`train.train`](../src/swarmflow/train.py#L228), [`curriculum.build_showcase`](../src/swarmflow/curriculum.py#L178).

## Repository evidence

[research plan](../docs/research-plan.md), [training reading list](../training.md), [showcase tests](../tests/test_showcase.py).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
