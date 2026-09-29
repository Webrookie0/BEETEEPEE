# Paper 06: Reciprocal n-body Collision Avoidance

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Jur van den Berg, Stephen J. Guy, Ming Lin and Dinesh Manocha. **Reciprocal n-body Collision Avoidance**. ISRR proceedings, 2011.

[Paper / official source](https://gamma-web.iacs.umd.edu/ORCA/)

**Relevant source sections:** Official description of velocity obstacles, reciprocal responsibility and optimization over permitted velocities; publication listing.

**Project status:** Relative-motion motivation for a sampled predictive heuristic; ORCA itself is not implemented.

**Verification:** The official ORCA project description and publication listing were checked on 29 September 2026. The project uses relative-motion reasoning, not the ORCA solver. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Relative-motion reasoning

ORCA reasons about collision-producing relative velocities and splits avoidance responsibility between cooperative agents. SwarmFlow uses measured relative positions/velocities to predict future clearances. External moving objects are treated as continuing their motion; no reciprocal cooperation is assumed.

For observing drone velocity $v_i$, a row's relative velocity gives $v_j=\Delta v_{ij}+v_i$. With a candidate own displacement $P_k(t)$:

$$g_{kj}(t)=\|P_k(t)-(\Delta p_{ij}+t(\Delta v_{ij}+v_i))\|-r_i-r_j.$$

Code: [`baseline.PredictiveController.action`](../src/swarmflow/baseline.py#L19). The full candidate costs are in [sampled planning](project_context.md#predictive-controller).

## Swept collision geometry

The independently implemented geometry checks minimum relative distance over a substep, rather than endpoint distance alone. For relative start $r_0$, end $r_1$ and $d=r_1-r_0$:

$$\alpha=\operatorname{clip}\left(-\frac{r_0\cdot d}{\max(d\cdot d,10^{-15})},0,1\right),\qquad
g_{min}=\|r_0+\alpha d\|-r_i-r_j.$$

Code: [`geometry.swept_distance`](../src/swarmflow/geometry.py#L8), [`env.SwarmEnv._clearance`](../src/swarmflow/env.py#L261). This elementary closest-point formula is project geometry, not an ORCA constraint or proof. Objects active for only part of the substep clip the tested time interval to their active interval.

## Differences and scope

There are no ORCA half-plane/half-space constraints, reciprocal velocity correction, linear program, or ORCA/RVO library calls. The baseline minimizes a soft sampled cost and may choose an unsafe velocity. Its parameter named `hard_margin` supplies a high penalty, not a hard feasibility constraint. It has no formal collision-free guarantee. Actual collision decisions are made by the environment after executing the finite-response motion.

## Repository evidence

[Research plan](../docs/research-plan.md), [baseline implementation](../src/swarmflow/baseline.py), [geometry](../src/swarmflow/geometry.py), and the high-speed collision test in [test_environment.py](../tests/test_environment.py).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
