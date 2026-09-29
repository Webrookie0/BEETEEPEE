# Paper 09: Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Christian Schroeder de Witt et al.. **Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?**. 2020.

[Paper / official source](https://arxiv.org/abs/2011.09533) · [Full text](https://arxiv.org/html/2011.09533)

**Relevant source sections:** §4 Independent PPO; §5.2 role of centralized value functions; §5.3 ablations.

**Project status:** Default algorithm-family choice: parameter-sharing independent PPO with a local critic.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Independent PPO

Each drone's actor and critic consume its own observation:

$$a_i\sim\pi_\theta(\cdot\mid o_i),\qquad V_i=V_\phi(o_i).$$

"Independent" refers to local learning inputs. The simulator still couples agents through collisions and formation motion, and all drones share one set of actor weights. Their samples jointly optimize the [PPO loss](paper_07.md#clipped-surrogate). The default critic has an independent encoder from the actor; "separate critic" and "centralized critic" are different choices.

The source establishes a useful baseline in SMAC, not an equivalence theorem or drone performance guarantee. SwarmFlow does not implement SMAC, recurrent memory or the paper's full experiments. `--critic central` switches to the optional [team-pooled value](paper_04.md#optional-centralized-value-function).

Code: [`policy.ActorCritic.value_local`](../src/swarmflow/policy.py#L113), [`policy.ActorCritic._critic_hidden`](../src/swarmflow/policy.py#L125), [`train.train`](../src/swarmflow/train.py#L228) (`critic='local'`, `separate_critic=True`).

## Repository evidence

[training §7](../training.md#7-what-was-used-from-which-paper), [knowledge §5](../knowledge.md#5-research-papers-what-we-took-from-each).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
