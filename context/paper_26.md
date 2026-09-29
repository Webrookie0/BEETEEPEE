# Paper 26: Deep Reinforcement Learning at the Edge of the Statistical Precipice

[Back to implementation sequence](README.md) · [Notation](project_context.md#notation)

## Source

Rishabh Agarwal et al.. **Deep Reinforcement Learning at the Edge of the Statistical Precipice**. NeurIPS 2021.

[Paper / official source](https://arxiv.org/abs/2108.13264) · [Full text](https://arxiv.org/html/2108.13264)

**Relevant source sections:** §4.1 stratified bootstrap confidence intervals; §4.2 performance profiles; §4.3 robust aggregate metrics; appendix A.4 maximum-based protocols.

**Project status:** Careful evaluation motivation; bootstrap/IQM/performance-profile toolbox not implemented.

**Verification:** Source identity and relevant full-text material were checked against the arXiv page on 29 September 2026. The equations use consistent project notation; they are not presented as verbatim paper transcriptions. Numerical settings belong to SwarmFlow unless stated otherwise.

## Uncertainty and complete reporting

The source argues for uncertainty and robust aggregation when RL results use few runs. SwarmFlow implements a simpler per-scenario normal-approximation interval for episode slot RMSE:

$$CI_{95}=\bar x\pm1.96\,s/\sqrt n\quad(n\ge2).$$

For one episode, `slot_rmse_std_m=0` and the interval is `null`. This is not the paper's stratified bootstrap and is not automatically reliable with three episodes, correlated seeds or skewed failures. There is no interquartile mean, probability-of-improvement computation or performance-profile implementation.

## Failure and recovery denominators

$$success=1-F/n,\qquad recovery_{5s}=E_{successful,\le5s}/E_{encountered}.$$

Failed episodes remain in failure/clearance/slot aggregates; their events do not qualify as recovered. Mean recovery time is conditional on a successful episode and a non-null recovery time, so it must be shown alongside recovery counts. Events not reached because an episode failed before object spawn are absent from the encountered-event denominator; zero failures is therefore a separate acceptance requirement.

The research plan's "rule of three" is an approximate bound for zero observed failures in independent trials: $p_{upper}\approx3/n$ at 95%; the exact one-sided binomial value is $1-0.05^{1/n}$. The code does not compute this bound. It is general statistical context, not a specific implemented method from this paper.

Code: [`evaluate.aggregate`](../src/swarmflow/evaluate.py#L37), [`evaluate.learning_checks`](../src/swarmflow/evaluate.py#L79). Current showcase uses 20 episodes per scenario; the original research target of 100 held-out episodes per scenario remains a separate proposed standard.

## Repository evidence

[research plan acceptance targets](../docs/research-plan.md), [evaluate.py](../src/swarmflow/evaluate.py), [training reading list](../training.md).

The [implementation sequence](README.md#implementation-sequence) links directly to the sections above. See [history and discrepancies](history.md) for older settings and [file inventory](file_inventory.md) for every retained artifact.
