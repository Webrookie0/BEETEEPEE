# Paper 21: Champion-level drone racing using deep reinforcement learning

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Elia Kaufmann et al.. **Champion-level drone racing using deep reinforcement learning**. Nature 2023.

[Paper / official source](https://www.nature.com/articles/s41586-023-06419-4)

**Relevant source sections:** Article overview and methods describing simulation-trained racing control and physical racing evaluation.

**Project status:** Background and training-scale comparison only.

**Verification:** The Nature article page and title were checked on 29 September 2026. It supplies background and scale context, not an implemented algorithm here. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Budget context

The project cites Swift to explain that successful drone RL can require substantial simulation and specialized training. No racing reward, gate perception, racing policy, simulator calibration or real-world control method from the paper is implemented here. No paper-specific equation is used in the runtime.

The useful project conversion is

$$\text{agent transitions}=N\times\text{team environment steps}.$$

Current pipeline v4 budgets sum to 3,000,000 PPO environment steps, or nominally 24,000,000 drone transitions for eight drones, excluding cloning and evaluation. Earlier 5 M/40 M figures describe older pipelines. A stage may end a few steps beyond its target because a batched tick advances all environments. The paper's interaction count is not a reliable laptop wall-clock estimate for this different task.

Code: [`train.train`](../src/swarmflow/train.py#L228) (step accounting), [curriculum configuration](configuration.md#curriculum-stages). No claim is made to reproduce champion-level racing or physical flight.

## Repository evidence

[training budget discussion and reading list](../training.md), [knowledge history](../knowledge.md).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
