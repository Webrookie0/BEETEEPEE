# Paper 04: The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Chao Yu et al.. **The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games**. NeurIPS 2022 Datasets and Benchmarks.

[Paper / official source](https://arxiv.org/abs/2103.01955) · [Full text](https://arxiv.org/html/2103.01955)

**Relevant source sections:** §3.2 MAPPO and IPPO; §3.3 implementation details; §§5.1–5.5 normalization, value inputs, data reuse, clipping and batch size; appendix C.2 parameter sharing.

**Project status:** Cooperative PPO design reference; default local critic, optional mean-pooled central critic; paper value normalization not implemented.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Cooperative PPO and parameter sharing

All drones contribute samples to one parameter-shared policy. The default uses each drone's local value $V_\phi(o_i)$, as in [IPPO](paper_09.md#independent-ppo). PPO clipping, limited epochs and large batches are appropriate controlled choices, not evidence of drone performance from the paper's game benchmarks.

Pipeline v4 uses $8\times256\times8=16{,}384$ agent transitions per full update, minibatches of 2,048 agent transitions, five epochs maximum, $\epsilon=0.2$ and approximate-KL stopping at 0.03. Exact update equations are in [PPO](paper_07.md#clipped-surrogate).

## Optional centralized value function

For a team of $N$ drones, let $h_i=E_\phi(o_i)$. With `--critic central`:

$$\bar h=\frac1N\sum_i h_i,\qquad V_i=f_\phi(\bar h)\quad\text{for all }i.$$

The team gets one shared value prediction expanded to all its drones. Actor inputs remain local. Minibatching preserves complete teams: shuffled units are environment/time groups, and each group expands to its $N$ contiguous rows. With the separate critic, its encoder is independent of the actor encoder.

Code: [`policy.ActorCritic.value_central`](../src/swarmflow/policy.py#L116), [`train.train`](../src/swarmflow/train.py#L228). This simple central critic is MAPPO-style, not a replication of MAPPO's full state input, agent-specific value conditioning, death masks or all optimizer details.

## Normalization distinction

The paper standardizes value targets with running mean and standard deviation and denormalizes predictions for GAE. SwarmFlow instead divides rewards by a running standard deviation of discounted returns:

$$\tilde r=r/\sqrt{\operatorname{Var}(G^{running})+10^{-8}}.$$

There is no mean-subtracted value-target normalizer or PopArt mechanism. Older notes saying "value normalization" should be read as motivation; the implemented operation is [reward scaling](paper_11.md#reward-scaling).

## Repository evidence

[training §7](../training.md#7-what-was-used-from-which-paper), [knowledge §5](../knowledge.md#5-research-papers-what-we-took-from-each), [test_learning.py](../tests/test_learning.py).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
