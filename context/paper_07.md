# Paper 07: Proximal Policy Optimization Algorithms

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

John Schulman et al.. **Proximal Policy Optimization Algorithms**. 2017.

[Paper / official source](https://arxiv.org/abs/1707.06347) · [PDF](https://arxiv.org/pdf/1707.06347)

**Relevant source sections:** Clipped surrogate objective and actor–critic training (source §§3 and 5, including Algorithm 1).

**Project status:** Implemented clipped PPO with project-specific scaling, stopping, imitation loss and continuous action distribution.

**Verification:** Source identity was checked against the arXiv abstract page on 29 September 2026. HTML full text was unavailable; the implemented equations were checked directly against the code. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Clipped surrogate

For stored old log likelihood $\ell_{old}$ and current $\ell_\theta$ of the same sampled action:

$$\rho_t=\exp(\ell_\theta-\ell_{old}),\qquad
L_{policy}=-\operatorname{mean}\left[\min\left(\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right].$$

Here $\epsilon=0.2$ by default. Advantages are standardized across the **whole collected batch**, with population standard deviation clamped to at least $10^{-8}$. Fresh policy rollouts are collected after each update; minibatches reuse only that update's rollout for at most five epochs in the curriculum. This is the paper's clipped surrogate adapted to shared multi-agent training. Clipping limits the objective's incentive; it does not strictly bound policy changes or prove safety.

Code: [`train.train`](../src/swarmflow/train.py#L228). Advantages come from [Paper 8](paper_08.md#gae-recurrence).

## Combined loss

The implementation first defines

$$L_V=\tfrac12\operatorname{mean}(V_\phi-\hat R)^2.$$

It then minimizes

$$L=L_{policy}+0.5L_V-c_H H(\mathcal N)+\beta(f)L_{BC}.$$

Thus the coefficient on the raw value squared error is **0.25**, not 0.5. The imitation term exists only when a teacher dataset is available and its coefficient is positive. Curriculum entropy coefficient $c_H=0$. During critic warm-up, only $0.5L_V$ is optimized and actor gradients are removed. No value-loss clipping is present.

## Squashed Gaussian actions

With three independent latent components:

$$u\sim\mathcal N(\mu_\theta(o),\operatorname{diag}(\sigma^2)),\quad a=\tanh u,\quad
\sigma=\exp(\operatorname{clamp}(\text{log\_std},m,1)).$$

The stored action for likelihood evaluation is the **raw latent** $u$. Change of variables gives

$$\log\pi(a\mid o)=\sum_{d=1}^3\left[\log\mathcal N(u_d;\mu_d,\sigma_d)
-2(\log2-u_d-\operatorname{softplus}(-2u_d))\right].$$

The correction is the numerically stable $\log(1-\tanh^2u)$. Deterministic evaluation uses $a=\tanh\mu$, which is the squashed latent mean, not generally the mathematical mean of the transformed distribution. Entropy uses the latent Gaussian proxy

$$H(\mathcal N)=\sum_d\tfrac12\log(2\pi e\sigma_d^2).$$

The environment further vector-clips $3a$ to 3 m/s; the policy density is defined for the normalized action before that environment mapping. Tanh distribution details are continuous-control engineering, not an assertion that the original PPO paper uses this exact implementation.

Code: [`policy.ActorCritic.sample`](../src/swarmflow/policy.py#L162), [`policy.ActorCritic.log_probability`](../src/swarmflow/policy.py#L158), [`policy.ActorCritic.evaluate`](../src/swarmflow/policy.py#L167), [`env.SwarmEnv.step`](../src/swarmflow/env.py#L307).

## Approximate KL and stopping

$$\widehat{D}_{KL}=\operatorname{mean}\left[(\rho-1)-\log\rho\right].$$

After each completed epoch, the code averages minibatch KL values and stops additional epochs when the mean exceeds `target_kl=0.03`, except during critic warm-up. It does not undo updates already made. Learning rate decays to 10% of its initial value, and gradients are clipped; see [Paper 11](paper_11.md#optimizer-and-schedules) and [implementation reference](references.md#ppo-implementation-details).

## Repository evidence

[training §7](../training.md#7-what-was-used-from-which-paper), [train.py](../src/swarmflow/train.py), [policy.py](../src/swarmflow/policy.py), and likelihood/reload tests in [test_learning.py](../tests/test_learning.py).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
