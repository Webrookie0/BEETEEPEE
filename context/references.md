# Other references and source-verification notes

[Implementation sequence](README.md) · [All papers](README.md#paper-index)

Research papers are numbered individually in `paper_01.md`–`paper_30.md`. This page preserves books, tutorials, documentation and source-code references without mislabelling them as research papers.

## PPO implementation details

Shengyi Huang et al., **The 37 Implementation Details of Proximal Policy Optimization**, ICLR Blog Track 2022. [Article](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/). Source sections checked: 13 core details, nine continuous-action details and auxiliary details.

Used ideas: batched collection across multiple environments; Adam epsilon 1e-5; normalized advantages; gradient clipping; learning-rate annealing; value/actor network design; diagnostic KL/entropy/explained variance. Code: [`train.train`](../src/swarmflow/train.py#L228), [`train.Worker`](../src/swarmflow/train.py#L64), [`policy.ActorCritic`](../src/swarmflow/policy.py#L86).

Exact current formulas are in [PPO](paper_07.md), [GAE](paper_08.md) and [scaling/schedules](paper_11.md). Project differences include whole-rollout advantage normalization rather than minibatch-only normalization, separate actor/critic gradient clipping, a tanh-squashed Gaussian with corrected likelihood, no value clipping and a separate teacher MSE term. No Atari preprocessing, LSTM, MultiDiscrete-action or asynchronous EnvPool implementation is present.

## Reinforcement learning background

- **Sutton and Barto, Reinforcement Learning: An Introduction, 2nd edition, 2018.** [Book page](http://incompleteideas.net/book/the-book-2nd.html). Existing reading list points to chapters 1, 3 and 13 for agents, returns, MDPs and policy gradients.
- **Spinning Up introduction and PPO tutorial.** [Documentation](https://spinningup.openai.com/en/latest/). Existing reading list uses it to explain policies, values, advantages and PPO. It is not a dependency or a copied trainer.

The shared background equations are $G_t=\sum_{k\ge0}\gamma^kr_{t+k}$, $V^\pi(o)=\mathbb E[G_t\mid o_t=o]$ as the project's observation-based value approximation, and $A^\pi=Q^\pi-V^\pi$ in the fully specified state/action formulation. The actual partially observed local policy estimates values from its current observation without recurrent state. PPO uses the finite-sample [GAE](paper_08.md#gae-recurrence) approximation.

## Gymnasium and PettingZoo

- [Gymnasium Hopper](https://gymnasium.farama.org/environments/mujoco/hopper/): the repository cites the healthy/alive reward as motivation for a positive per-second alive term. SwarmFlow's reward is $\Delta t(1-C)$ minus culprit failure penalties; it does not import Hopper rewards, MuJoCo dynamics or Hopper termination rules.
- [PettingZoo](https://pettingzoo.farama.org/): `SwarmEnv` subclasses `ParallelEnv`; reset returns observations/infos; step returns observations/rewards/terminations/truncations/infos dictionaries keyed by drone. Every active agent supplies an action simultaneously; episode end clears `agents`.
- Gymnasium `spaces.Dict` and `spaces.Box` define the observation/action tensors. The environment performs its own action validation.

These are software/documentation references, not additional research papers inferred from dependency names. Code: [`env.SwarmEnv.observation_space`](../src/swarmflow/env.py#L41), [`env.SwarmEnv.action_space`](../src/swarmflow/env.py#L48), [`env.SwarmEnv.step`](../src/swarmflow/env.py#L307). See [Paper 12](paper_12.md) for temporal semantics.

## Source code and tooling references

- [quad-swarm-rl](https://github.com/Zhehui-Huang/quad-swarm-rl/) accompanies [QuadSwarm](paper_03.md). It is a design/hyperparameter reference, not the base implementation, and the historical comparison is not pinned to a commit.
- [uv installation](https://docs.astral.sh/uv/getting-started/installation/) is linked by the launcher/setup instructions.
- Exact runtime dependencies and supported versions are in [configuration](configuration.md#dependencies-and-build) and the retained [lockfile](../uv.lock). Optimizer and neural-layer operations come from PyTorch, not from locally reimplemented external algorithms.

## Verification and scope

All 25 arXiv identifiers present in the original project Markdown were checked against their abstract-page titles on 29 September 2026. HTML full text was available for 24 of those papers, allowing relevant methods/section headings to be checked; PPO's abstract and PDF link were available but its HTML full text was not. The ORCA project page, Nature article page and PPO implementation blog were also checked. The three publisher links for Bengio curriculum learning, Mayne robust MPC and Williams information-theoretic MPC did not provide readable full text in this session; those notes explicitly mark the limitation and avoid invented source equation/section numbers.

Numbered source sections refer to the accessible paper version. Numbered **README implementation sections** are a separate navigation order. Local links land on project explanations of the relevant paper method; source links then lead to the paper itself. The notes summarize methods in original wording and include only relevant mathematical expressions, not copies of complete copyrighted papers.

The original repository contains no research-paper PDFs or complete bibliography database. Most paper links were concentrated in research-plan.md, training.md and knowledge.md; notes warned that some prior comparison details were recalled from memory. Here, code-derived equations and settings take priority over those older summaries. Unavailable legacy source code and missing experiment artifacts are listed in [history](history.md#missing-and-historical-only-context).

## Updating these notes

When the code or recipe changes, update the corresponding equation/default and source-function link, the relevant paper status, the implementation sequence, and affected history claims. When adding a paper, use a new number rather than reusing an existing one; specify its contribution and whether its actual algorithm is implemented. Preserve old experiment protocol/version labels. Regenerate or update the [file inventory](file_inventory.md) when adding artifacts. Check local file paths and heading fragments after changing headings.
