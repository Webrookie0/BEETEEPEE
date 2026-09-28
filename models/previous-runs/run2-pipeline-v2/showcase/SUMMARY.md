# Showcase model

- Model: `best.pt`, copied from `models/stage3-seed0/best.pt`
- Trained through: stage1-seed0 -> stage2-seed0 -> stage3-seed0
- Chosen at 1,701,888 environment steps with validation score 0.861 (selection seeds 5000+)
- Held-out test seeds: 100000..100019; test score learned 0.815 vs baseline 0.962
- **Learned correctly: NOT YET**

| Stage | Scenarios | Steps at end | Best validation score | Best at step |
| --- | --- | --- | --- | --- |
| Stage 1 | hold, follow, turn | 1,000,000 | 0.996 | 901,120 |
| Stage 2 | hold, follow, turn, crossing, head_on | 2,901,120 | 0.956 | 1,501,184 |
| Stage 3 | hold, follow, crossing, head_on, double_crossing, slalom, turn, gust | 3,501,184 | 0.861 | 1,701,888 |

| Check | Result | Detail |
| --- | --- | --- |
| No collisions or other failures | FAIL | 148/160 episodes completed |
| >= 95% of disturbances recovered within 5 s | FAIL | 66/80 events |
| Slot RMSE below 0.35 m when undisturbed | PASS | hold 0.080 m, follow 0.052 m, turn 0.080 m |

| Scenario | Success (baseline / learned) | Slot RMSE m (baseline / learned) | Recovered events (baseline / learned) |
| --- | --- | --- | --- |
| hold | 1.000 / 1.000 | 0.014 / 0.080 | 0/0 / 0/0 |
| follow | 1.000 / 1.000 | 0.014 / 0.052 | 0/0 / 0/0 |
| crossing | 1.000 / 1.000 | 0.195 / 0.257 | 20/20 / 20/20 |
| head_on | 1.000 / 0.900 | 0.182 / 0.332 | 20/20 / 18/20 |
| double_crossing | 1.000 / 0.900 | 0.272 / 0.420 | 40/40 / 32/40 |
| slalom | 1.000 / 0.600 | 0.393 / 1.170 | 0/0 / 0/0 |
| turn | 1.000 / 1.000 | 0.033 / 0.080 | 0/0 / 0/0 |
| gust | 1.000 / 1.000 | 0.128 / 0.140 | 0/0 / 0/0 |

View it: `python -m swarmflow view` (loads this model automatically).
