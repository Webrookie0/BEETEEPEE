# Paper 10: What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Marcin Andrychowicz et al.. **What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study**. ICLR 2021.

[Paper / official source](https://arxiv.org/abs/2006.05990) · [Full text](https://arxiv.org/html/2006.05990)

**Relevant source sections:** §3.2 network architecture; §3.3 normalization/clipping; §3.5 training setup; §3.7 optimizers; appendix network/action distribution choices.

**Project status:** Implementation recommendations adapted to this controller.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Separate networks and initialization

The actor and default critic have separate encoders so value-loss gradients cannot alter the actor's observation features. Only the actor output weight is explicitly orthogonally initialized with gain 0.01; its bias is zero. Other layers use their PyTorch defaults. Older prose implying every layer is orthogonally initialized overstates the code.

$$\mu(o)=W_a E_\theta(o)+b_a,\qquad W_a\leftarrow0.01\,\operatorname{Orthogonal},\quad b_a=0.$$

The width is 128 in curriculum runs, 64 in direct `train()` defaults. Tanh MLP activations and the [squashed Gaussian](paper_07.md#squashed-gaussian-actions) are used. The parameter count is architecture-dependent; the metadata, rather than a rounded historical count, specifies the network.

## Exploration settings

When cloning is requested and no override is supplied, initial `log_std=-1.6`, so $\sigma\approx0.202$ in latent normalized-action units. Without cloning it starts at $e^{-1}\approx0.368$. Pipeline minimum log standard deviation is $-3$, giving a floor of approximately 0.0498; upper clamp is 1. After resumed-stage cloning, `log_std` is raised **at least** to the requested initialization using `clamp_(min=...)`, not forcibly reset to exactly that value.

Multiplying 0.202 by 3 gives a local small-signal velocity scale of about 0.61 m/s, but tanh and norm clipping make it state-dependent. Entropy bonus is zero in current defaults. These are project settings motivated by empirical implementation studies; they are not globally optimal constants.

## Actor and critic gradients

Separate networks receive separate norm clipping at 0.5; shared-critic mode clips the combined model norm. Critic-only warm-up clears actor gradients before optimizer steps. See [optimizer equations](paper_11.md#optimizer-and-schedules).

Code: [`policy.ActorCritic.__init__`](../src/swarmflow/policy.py#L90), [`policy.ActorCritic.actor_parameters`](../src/swarmflow/policy.py#L148), [`policy.ActorCritic.critic_parameters`](../src/swarmflow/policy.py#L140), [`train.train`](../src/swarmflow/train.py#L228). `test_separate_critic_does_not_touch_actor_weights` checks the intended separation.

## Repository evidence

[training diagnosis and §7](../training.md), [policy.py](../src/swarmflow/policy.py), [test_learning.py](../tests/test_learning.py).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
