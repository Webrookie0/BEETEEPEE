# Paper 28: Robust Model Predictive Control of Constrained Linear Systems with Bounded Disturbances

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

David Q. Mayne, María M. Seron and Saša V. Raković. **Robust Model Predictive Control of Constrained Linear Systems with Bounded Disturbances**. Automatica 2005.

[Paper / official source](https://doi.org/10.1016/j.automatica.2004.08.019)

**Relevant source sections:** Robust constraint tightening / allowance for bounded tracking disturbances. Exact publisher section numbering was not verified.

**Project status:** Qualitative motivation for a larger teacher clearance margin; robust/tube MPC not implemented.

**Verification:** The source link is retained from the project bibliography; publisher full text was not available during this documentation pass. Equations below describe the project code unless explicitly marked otherwise. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Clearance margin for imitation error

The project uses the robust-control intuition that nominal plans should leave space for tracking error. A conceptual allowance is

$$g_{planned}\ge g_{required}+\varepsilon_{tracking}.$$

This inequality explains the design choice; the code neither proves nor estimates a uniform error bound $\varepsilon_{tracking}$. Instead it changes the predictive teacher's soft/hard penalty thresholds from baseline 0.35/0.08 m to **0.7/0.3 m** in pipeline v4 (v3 used 0.6/0.25 m).

$$C_{avoid}=180\,[m_s-g_{min}]_+^2+20000\,[m_h-g_{min}]_+^2.$$

This expression is applied separately to neighbor and obstacle groups, taking the worst predicted gap in each group. Full details are [in the planner](project_context.md#predictive-controller). Even the "hard" margin is a penalty, not a guaranteed constraint.

## Scope of the adaptation

No disturbance invariant set, tightened feasible set, tube controller, terminal robust invariant set, linear plant proof or robust MPC optimization is implemented. Larger margins may improve clearance but increase detours and recovery time. The historical experiments support an empirical choice, not formal safety or an identified bound on neural imitation error.

Code: [`baseline.PredictiveController.__init__`](../src/swarmflow/baseline.py#L10), [`baseline.PredictiveController.action`](../src/swarmflow/baseline.py#L19), [`train.clone`](../src/swarmflow/train.py#L184), [curriculum constants](../src/swarmflow/curriculum.py).

## Repository evidence

[training Part E](../training.md), [knowledge §5.3](../knowledge.md#53-papers-added-in-later-versions).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
