# Paper 22: Curriculum Learning

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Yoshua Bengio, Jérôme Louradour, Ronan Collobert and Jason Weston. **Curriculum Learning**. ICML 2009.

[Paper / official source](https://dl.acm.org/doi/10.1145/1553374.1553380)

**Relevant source sections:** Curriculum strategy: progressively change the training distribution from easier to harder examples. Exact publisher section numbering was not verified.

**Project status:** Easy-to-hard scheduling principle; fixed, hand-authored task stages.

**Verification:** The source link is retained from the project bibliography; publisher full text was not available during this documentation pass. Equations below describe the project code unless explicitly marked otherwise. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Progressive task distribution

SwarmFlow changes the scenario set by stage, retaining earlier scenarios:

$$\mathcal S_1=\{hold,follow,turn\},$$
$$\mathcal S_2=\mathcal S_1\cup\{crossing,head\_on\},$$
$$\mathcal S_3=\mathcal S_2\cup\{double\_crossing,gust\}.$$

Each new episode samples a scenario uniformly from its stage set. The next stage loads the previous stage's best validated checkpoint, performs new cloning/DAgger, then runs its fixed PPO budget. This is the project's concrete curriculum, not a schedule derived from an equation in the source.

## Implementation limits

All three stages currently allocate 1 M PPO team steps. Promotion is budget/completion driven, not automatic mastery or a validation threshold; the research plan's proposed validation-gated progression is not implemented. Retaining old tasks offers practice but does not prove prevention of forgetting. Dense, slalom, pilot, terrain and non-default formation geometries are not included in default curriculum training.

Code: [`curriculum.stage_arguments`](../src/swarmflow/curriculum.py#L62), [`curriculum.train_stage`](../src/swarmflow/curriculum.py#L75), [`curriculum.run_pipeline`](../src/swarmflow/curriculum.py#L127). Exact settings: [curriculum stages](configuration.md#curriculum-stages).

## Repository evidence

[research plan: ordered execution](../docs/research-plan.md), [training §6 and version history](../training.md), [guide](../guide.md).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
