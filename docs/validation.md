# Development validation — 26 September 2026

These measurements use the reduced-order stabilized-flight model and the parameters recorded in each report. They are development checks, not publication results or physical-flight validation. The baseline was refined using the development trials; these seeds are not an untouched final test set.

## Automated checks

`python -m pytest -q`: **18 passed**. Coverage includes the PettingZoo parallel API, deterministic reset/step behavior, immutable target-relative slot assignments, high-speed swept collision detection, command validation, acceleration limits, visibility masks, full-duration recovery with per-drone slot checks, the no-avoidance comparison, policy permutation/padding invariance, consistent squashed action probabilities, time-limit GAE, actual optimization, checkpoint reload and optimizer resume.

The Python modules compile and the CLI exposes `view`, `eval` and `train`.

## Eight-drone baseline

[Full raw report](../runs/baseline.json). Three episodes per scenario, seeds 1000, 1001 and 1002; 24 simulated seconds per completed episode. All **24/24 episodes** completed without collision or boundary failure.

- Hold: mean episode slot RMSE 0.0144 m.
- Follow: mean episode slot RMSE 0.0144 m.
- Crossing: mean episode slot RMSE 0.2011 m; 3/3 events recovered; mean recovery 1.018 s.
- Head-on: mean episode slot RMSE 0.1845 m; 3/3 events recovered; mean recovery 1.033 s.
- Repeated crossing: mean episode slot RMSE 0.2676 m; 6/6 events recovered; mean recovery 1.018 s.
- Static spheres: mean episode slot RMSE 0.3775 m.
- Turning target: mean episode slot RMSE 0.0335 m.
- Wind gust: mean episode slot RMSE 0.1286 m.

Smallest measured surface clearance across these episodes: **0.0823 m**. All **12/12 encountered recovery events** succeeded. Recovery time includes one full second of confirmed settling after the object leaves the modeled scene; it does not measure the entire duration of the avoidance maneuver.

This is a small, easy-to-reproduce starting suite. Target motion and most event geometry are scripted, sensing is ideal in these runs, and three seeds per scenario provide weak statistical evidence. The low clearance leaves little room for unmodeled errors. Larger noisy tests and full flight dynamics remain necessary.

## Controls and scaling

[No-avoidance report](../runs/no-avoidance.json): the slot-only controller collided in **9/9 episodes**, across crossing, head-on and repeated-crossing scenarios with the same three seeds. Its smaller slot errors reflect refusal to avoid obstacles and shorter failed episodes; they are not better overall performance.

[16-drone report](../runs/scaling-16.json): crossing and head-on scenarios, seeds 2000 and 2001, **4/4 episodes** without collision, **4/4 events** recovered. Minimum surface clearance 0.1371 m. This checks variable team size in two scenarios; it does not validate all team sizes or all densities.

## Learning pipeline

[Smoke log](../runs/smoke-policy/training.jsonl), [checkpoint](../runs/smoke-policy/latest.pt), [evaluation](../runs/smoke-policy/evaluation.json).

The smoke run collected 128 baseline environment steps for cloning (1024 drone examples), used three cloning epochs, and ran 512 PPO environment steps (4096 agent transitions), with 64-step rollouts and two update epochs. It saved and reloaded the policy, produced finite losses, and measured roughly 1100 environment steps/s in this short local CPU run. This is not an estimate for a physics simulator or a statistically stable throughput benchmark.

The checkpoint is **not qualified**: it completed two follow episodes but drifted to approximately 2.85–2.88 m slot RMSE, and collided in both crossing evaluations. Successful baseline behavior must not be attributed to this neural policy. Actual multi-seed curriculum training and validation are the next experiment. The final trainer explicitly samples scenario seeds 0–999; default evaluation uses 1000 onward. The earlier smoke run was produced before that seed-range restriction and is only a machinery check.

## Interactive verification

Verified in Chrome against the local Python server:

- Start, finish and reset the scripted crossing scenario; observed 1/1 recovered events and approximately zero final slot error.
- Switch to target following, inject a crossing object manually, run the episode; observed 1/1 recovered events, no collision, and approximately zero final slot error.
- Inspect orbit rendering, persistent drone labels, slot markers, trails and telemetry.
- Export the view as a PNG; [saved preview](flight-preview.png).
- No warning/error entries in the browser's captured console during the tested flight flow.

Only the desktop layout has been visually checked. A mobile stylesheet exists but has not been independently validated. The viewer is an orthographic canvas projection of 3D state; its displayed drone attitude is illustrative.

## Pipeline additions

Makefile targets now include `train-stage1-multi` / `eval-stage1-multi` / `train-stage2-multi` / `eval-stage2-multi` (`SEEDS` default `0 1 2`, outputs `runs/stage1-seed$(S)`), `eval-heldout` (seed start 100000, `SEEDS_N` default 20), `eval-noisy` (`--noise 0.05 --dropout 0.1`), and `eval-dense` (`dense` and `slalom`, 12 obstacle slots). Evaluation JSON summaries add success rate and slot-RMSE confidence intervals; `swarmflow compare` diffs two reports. These targets and fields were not used for the historical numbers above.
