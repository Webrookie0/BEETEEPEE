# Showcase model

- Model: `best.pt`, copied from `models/stage3-seed0/best.pt`
- Trained through: stage1-seed0 -> stage2-seed0 -> stage3-seed0
- Chosen at 501,760 environment steps with validation score 0.496 (selection seeds 5000+)
- Held-out test seeds: 100000..100019; test score learned 0.513 vs baseline 0.962
- **Learned correctly: NOT YET**

| Stage | Scenarios | Steps at end | Best validation score | Best at step |
| --- | --- | --- | --- | --- |
| Stage 1 | hold, follow, turn | 1,000,000 | 0.997 | 401,408 |
| Stage 2 | hold, follow, turn, crossing, head_on | 2,401,408 | 0.788 | 501,760 |
| Stage 3 | hold, follow, crossing, head_on, double_crossing, slalom, turn, gust | - | 0.496 | 501,760 |

| Check | Result | Detail |
| --- | --- | --- |
| No collisions or other failures | FAIL | 138/160 episodes completed |
| >= 95% of disturbances recovered within 5 s | FAIL | 38/80 events |
| Slot RMSE below 0.35 m when undisturbed | PASS | hold 0.152 m, follow 0.187 m, turn 0.187 m |

| Scenario | Success (baseline / learned) | Slot RMSE m (baseline / learned) | Recovered events (baseline / learned) |
| --- | --- | --- | --- |
| hold | 1.000 / 1.000 | 0.014 / 0.152 | 0/0 / 0/0 |
| follow | 1.000 / 1.000 | 0.014 / 0.187 | 0/0 / 0/0 |
| crossing | 1.000 / 1.000 | 0.195 / 1.360 | 20/20 / 20/20 |
| head_on | 1.000 / 0.900 | 0.182 / 1.106 | 20/20 / 18/20 |
| double_crossing | 1.000 / 0.000 | 0.272 / 1.799 | 40/40 / 0/40 |
| slalom | 1.000 / 1.000 | 0.393 / 2.627 | 0/0 / 0/0 |
| turn | 1.000 / 1.000 | 0.033 / 0.187 | 0/0 / 0/0 |
| gust | 1.000 / 1.000 | 0.128 / 0.196 | 0/0 / 0/0 |

View it: `python -m swarmflow view` (loads this model automatically).
