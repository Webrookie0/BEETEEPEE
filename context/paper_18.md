# Paper 18: Deep Sets

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Manzil Zaheer et al.. **Deep Sets**. NeurIPS 2017.

[Paper / official source](https://arxiv.org/abs/1703.06114) · [Full text](https://arxiv.org/html/1703.06114)

**Relevant source sections:** §2 permutation invariance and equivariance; §2.2 structure; §3.1 architecture.

**Project status:** Permutation-invariant set-processing principle implemented through attention pooling.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Order-independent neighbor sets

The source's canonical invariant representation is

$$f(X)=\rho\left(\sum_{x\in X}\phi(x)\right),\qquad f(PX)=f(X)$$

for a permutation $P$, under the representation theorem's assumptions. The key project requirement is that reordering visible neighbors or obstacles must not change a drone's action.

SwarmFlow embeds each valid row with the same MLP, then performs learned-query attention pooling rather than a literal sum. Neighbor and obstacle sets have distinct encoders; concatenating them with the self embedding preserves their different meanings. Missing rows are masked; a fully empty set returns a zero vector. Permutation invariance does not imply invariance to adding new visible objects or to changing density/capacity.

The exact [attention formula](paper_19.md#attention-pooling) is the implemented aggregation. Code: [`policy.SetEncoder.forward`](../src/swarmflow/policy.py#L46), [`policy.Encoder.encode`](../src/swarmflow/policy.py#L78). Tests permute rows, perturb masked padding and check equivalent outputs. No claim is made to reproduce every Deep Sets theorem or experiment.

## Repository evidence

[training §11.12](../training.md), [policy tests](../tests/test_learning.py).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
