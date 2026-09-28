# Showcase model

- Model: `best.pt`, copied from `models/stage3-seed0/best.pt`
- Trained through: stage1-seed0 -> stage2-seed0 -> stage3-seed0
- Chosen at 700,416 environment steps with validation score 0.957 (selection seeds 5000+)
- Held-out test seeds: 100000..100019; test score learned 0.936 vs baseline 0.970
- **Learned correctly: NOT YET**

| Stage | Scenarios | Steps at end | Best validation score | Best at step |
| --- | --- | --- | --- | --- |
| Stage 1 | hold, follow, turn | 1,000,000 | 0.997 | 700,416 |
| Stage 2 | hold, follow, turn, crossing, head_on | 2,700,416 | 0.968 | 700,416 |
| Stage 3 | hold, follow, crossing, head_on, double_crossing, turn, gust | 2,700,416 | 0.957 | 700,416 |

| Check | Result | Detail |
| --- | --- | --- |
| No collisions or other failures | FAIL | 138/140 episodes completed |
| >= 95% of disturbances recovered within 5 s | FAIL | 70/80 events |
| Slot RMSE below 0.35 m when undisturbed | PASS | hold 0.023 m, follow 0.027 m, turn 0.045 m |

| Scenario | Success (baseline / learned) | Slot RMSE m (baseline / learned) | Recovered events (baseline / learned) |
| --- | --- | --- | --- |
| hold | 1.000 / 1.000 | 0.014 / 0.023 | 0/0 / 0/0 |
| follow | 1.000 / 1.000 | 0.014 / 0.027 | 0/0 / 0/0 |
| crossing | 1.000 / 1.000 | 0.195 / 0.274 | 20/20 / 20/20 |
| head_on | 1.000 / 0.900 | 0.182 / 0.289 | 20/20 / 18/20 |
| double_crossing | 1.000 / 1.000 | 0.272 / 0.393 | 40/40 / 40/40 |
| turn | 1.000 / 1.000 | 0.033 / 0.045 | 0/0 / 0/0 |
| gust | 1.000 / 1.000 | 0.128 / 0.176 | 0/0 / 0/0 |

View it: `python -m swarmflow view` (loads this model automatically).
