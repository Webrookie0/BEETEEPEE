# Paper 11: Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Logan Engstrom et al.. **Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO**. ICLR 2020.

[Paper / official source](https://arxiv.org/abs/2005.12729) · [Full text](https://arxiv.org/html/2005.12729)

**Relevant source sections:** §3 attribution of PPO performance; §4 code-level optimizations; appendix A.2 PPO implementation optimizations.

**Project status:** Motivation for documented reward scaling, scheduling, clipping and initialization; exact choices are project-specific.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Reward scaling

`RewardScaler` accumulates a discounted trace separately for each environment/drone:

$$G_t^{run}=\gamma G_{t-1}^{run}+r_t,\qquad
\tilde r_t=r_t/\sqrt{v+10^{-8}}.$$

It updates running moments from the new traces before zeroing traces for ended episodes. It does **not** subtract a mean from rewards, clip scaled rewards, or normalize actor inputs online. Actor input scales are fixed physical-unit divisors.

For old count $n$, mean $\mu$, variance $v$ and a batch of count $m$, mean $\mu_b$, variance $v_b$:

$$\Delta=\mu_b-\mu,\quad n'=n+m,\quad\mu'=\mu+\Delta m/n',$$
$$v'=\frac{nv+mv_b+\Delta^2nm/n'}{n'}.$$

Initial count is $10^{-4}$, mean 0, variance 1. Checkpoints save these moments; per-worker discounted traces restart on resume. This is running reward scaling, distinct from [MAPPO value normalization](paper_04.md#normalization-distinction).

Code: [`train.RewardScaler.__call__`](../src/swarmflow/train.py#L45), [`train.RewardScaler._update`](../src/swarmflow/train.py#L53).

## Optimizer and schedules

For stage progress $f\in[0,1]$:

$$\eta(f)=\eta_0(1-0.9f),\qquad
\beta(f)=\beta_0+(\beta_1-\beta_0)f.$$

Learning rate ends at 10% of the initial rate. Current demonstration coefficient is 1.0 → 0.5. Adam uses `eps=1e-5`, default betas $(0.9,0.999)$, no configured weight decay. The standard optimizer operations are

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\quad
\theta_t=\theta_{t-1}-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon_{Adam}),$$

with bias-corrected moments. Adam is provided by PyTorch; this is dependency behavior, not an Adam invention attributed to Engstrom. Gradient norm clipping at 0.5 conceptually replaces $g$ by $g\min(1,0.5/\|g\|)$ (PyTorch includes a numerical epsilon). Initialization details are [here](paper_10.md#separate-networks-and-initialization).

## Diagnostics

$$EV=1-\frac{\operatorname{Var}(\hat R-V_{old})}{\operatorname{Var}(\hat R)}.$$

Explained variance is logged using the stored rollout values, before updates; it is `null` if return variance is at most $10^{-12}$. Logs also report raw reward, KL, latent entropy, action standard deviation, losses and throughput. These diagnostics do not replace held-out task evaluation. The paper's empirical lesson is to record implementation details and controls, not to attribute every improvement to the PPO name.

## Repository evidence

[training §7 and implementation details](../training.md), [train.py](../src/swarmflow/train.py), and [references: PPO blog](references.md#ppo-implementation-details).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
