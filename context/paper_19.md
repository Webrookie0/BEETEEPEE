# Paper 19: Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Juho Lee et al.. **Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks**. ICML 2019.

[Paper / official source](https://arxiv.org/abs/1810.00825) · [Full text](https://arxiv.org/html/1810.00825)

**Relevant source sections:** §2.2 attention; §3.2 pooling by multihead attention; §3.4 invariance analysis.

**Project status:** Learned-query multihead attention pooling, simplified from the full Set Transformer.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Attention pooling

For each neighbor/object row $x_j$ (excluding the mask), normalize by $[7,7,7,6,6,6,1.5]$ and embed with a two-layer tanh MLP, dimensions $7\to w/2\to w$ (10 inputs if optional gap features are enabled). For each attention head:

$$Q=qW_Q,\quad K_j=h_jW_K,\quad V_j=h_jW_V,$$
$$\alpha_j=\operatorname{softmax}_j(QK_j^\top/\sqrt{d_k}+M_j),\qquad
z_h=\sum_j\alpha_jV_j,$$
$$z=\operatorname{Concat}(z_1,\ldots,z_H)W_O.$$

$q$ is one learned query initialized to zero. $M_j$ excludes masked rows. PyTorch projection biases are also present. $H=\max(2,w//32)$: two heads at width 64, four at width 128. All-empty rows temporarily unmask keys to avoid undefined all-masked attention, then the pooled output is explicitly zeroed.

## Complete encoder

The self vector is scaled by $[5,5,5,3,3,3,3,3,3,3,3,3,5,5,5,10,1]$ and mapped $17\to w$ with tanh. Its features are concatenated with separate neighbor and object pools:

$$h=\operatorname{MLP}_{3w\to2w\to w}([h_{self},z_{neighbors},z_{obstacles}]).$$

The actor maps $h$ to three latent means. The local critic maps its own encoder output to a scalar. These fixed-size outputs permit variable visible-set sizes and drone counts without changing the actor's parameter shapes.

## Differences and checks

This is a PMA-inspired pooled representation, not a full SAB/ISAB encoder–decoder stack. It does not include all residual/layer-normalization blocks from the paper, and its learned query is not conditioned directly on self state. Self information joins after the pools. It supplies order invariance, not proven swarm-size generalization.

Code: [`policy.SetEncoder`](../src/swarmflow/policy.py#L30), [`policy.Encoder`](../src/swarmflow/policy.py#L68), [`policy.ActorCritic`](../src/swarmflow/policy.py#L86). Tests: `test_attention_encoder_permutation_invariance`, `test_policy_is_invariant_to_object_order_and_ignores_padding`, and the optional gap-feature round-trip test.

## Repository evidence

[training §11.12](../training.md), [implementation contract](../docs/implementation.md), [test_learning.py](../tests/test_learning.py).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
