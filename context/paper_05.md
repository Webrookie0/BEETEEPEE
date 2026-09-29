# Paper 05: Learning to Fly — a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Jacopo Panerati et al.. **Learning to Fly — a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control**. IROS 2021.

[Paper / official source](https://arxiv.org/abs/2103.02142) · [Full text](https://arxiv.org/html/2103.02142)

**Relevant source sections:** §III-B Bullet physics; III-C quadcopter dynamics/aerodynamic effects; III-E action spaces, especially desired velocity input; V-A control examples.

**Project status:** Future physics-validation reference, not an installed simulation backend.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Stabilized velocity interface

The source distinguishes physical quadrotor dynamics and controller-level action interfaces. SwarmFlow chooses a fast velocity-command abstraction to isolate coordination and recovery:

$$a^{cmd}=\operatorname{limit}_{a_{max}}((v^{des}-v)/\tau).$$

This formula is SwarmFlow's first-order stabilized response, **not** the paper's rigid-body or motor dynamics. See [flight dynamics](project_context.md#flight-dynamics) for substep integration and disturbances.

## Future physical validation

The documented future sequence is to reproduce the same trajectories in a full drone physics backend, calibrate velocity tracking lag, impose thrust/tilt limits, include downwash and parameter variation, and re-evaluate collisions and recovery. No PyBullet dependency, six-degree-of-freedom dynamics, aerodynamic model or hardware interface is present in [pyproject.toml](../pyproject.toml). No physical-model equation from this paper is currently executed.

## Repository evidence

[Research plan: direction and physics validation](../docs/research-plan.md), [implementation contract](../docs/implementation.md).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
