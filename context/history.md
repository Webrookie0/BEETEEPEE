# History, evidence and unresolved context

[Implementation sequence](README.md) · [All artifacts](file_inventory.md)

This is a documentation snapshot on 29 September 2026. Stored results are historical measurements, not evaluations rerun for this task. Current run folders may continue changing while training runs. Original documents and artifacts are preserved; the new context reconciles their claims against current code and available JSON evidence.

## Earlier BTP1 project

`results/archive` and `results/submit` come from the earlier Windows BTP1_v1 project. Their schema has `mode`, `policy`, `ckpt`, `seed`, `dumped`, `model_dump`, three pass/fail `gates` and `extra`. Windows paths inside JSON are provenance strings, not working paths in this checkout.

Legacy CSV metric columns are step, reward, collisions, score, t1_err/t2_err/t3_err and t1_pass/t2_pass/t3_pass. Trajectory CSVs contain t, test_id, drone, xyz, velocity xyz, acceleration xyz and target xyz. Images, distance graphs and copied checkpoints accompany many packs. These are data from a different evaluation contract; their gate error/score is not SwarmFlow slot RMSE/selection score.

`graphify-out/.graphify_labels.json` mentions an attitude-command design, geometric PID, quadrotor model, pass_gate.py, GazeboView, PythonView and older module documentation. The actual older dynamics/controller/gate source files are not present. Their equations, gains and training provenance cannot be reconstructed reliably from labels or binary checkpoints. No paper attribution is invented for them. Root-level `models/best.pt` and `models/latest.pt` belong to that legacy context according to guide.md and are ignored by current model discovery.

The submitted pack records three failed gates with errors approximately 45.206, 172.241 and 144.642, reason `hold_failed`. [Original submitted report](../results/submit/result.md) and [JSON](../results/submit/result.json) retain the exact values. The per-pack index below reads each available archived result, rather than assuming an old prose count.

## Initial SwarmFlow development

The research plan dated 26 September proposed the stabilized model, immutable slots, predictive baseline, local policy and separate collision/formation/recovery metrics. [docs/validation.md](../docs/validation.md) records a 24-episode baseline check, 9/9 slot-only collisions and four 16-drone checks. It also records a smoke network that drifted and collided. These are small development checks, not broad learned-policy results.

Early short runs under `runs/` used older reward/critic formats. The historical diagnosis cites unbounded negative costs, value losses dominating shared actor features, small datasets, unprepared critics and inadequate obstacle imitation. One illustrative old-cost example was an approximately −2.45 step reward at 7 m error, with discounted cost about −245 versus a −200 terminal penalty; it is a historical explanation, not the current reward equation. The v1 rebuild added bounded costs, alive reward, separate critic, noisy cloning/DAgger, scaling, warm-up, seed separation and best-checkpoint selection.

## Pipeline changes

- **v1:** stage 1 cloning only, width 64, initially team-wide failure penalties. Formation learning worked; obstacle stages could deteriorate. Saved checkpoint format for archived run 1 is v3, showing that pipeline and checkpoint version numbers differ.
- **v2:** cloning/DAgger in every stage; teacher loss inside PPO; exploration floor; culprit-only penalties; 1 m warning; failure-state restores; width 128; ten selection seeds; fine-tuning rate 1e-4. Teacher used 0.35/0.08 m and imitation faded to 0.1.
- **v3:** teacher margins 0.6/0.25 m; imitation end weight 0.5; slalom removed from learned benchmark; replayed crashes counted in failures. This changed the learned showcase from eight to seven scenarios. Scores/counts across that change are not a matched-protocol ablation.
- **v4:** teacher margins 0.7/0.3 m; overlap-aware recovery; stages 2–3 shortened from 2 M to 1 M PPO steps. Gap features and MPPI-style blending remain optional/off after negative development experiments; hard-state weighting defaults to 1. Current code does not contain a learned-policy safety filter.

Research rationale: [DAPG](paper_15.md), [Kickstarting](paper_27.md), [robust margin intuition](paper_28.md), [MPPI-style trial](paper_29.md), [forgetting](paper_30.md). Detailed historical experiments, including rejected settings, remain in [training.md Parts C–E](../training.md).

## Saved training runs

The following entries are read from each saved showcase summary. "Recovery within 5 s" comes from its stored check, not from reinterpreting the event times using today's overlap rule.

### run1-pipeline-v1

- Evidence: [models/previous-runs/run1-pipeline-v1/showcase/summary.json](../models/previous-runs/run1-pipeline-v1/showcase/summary.json) and [models/previous-runs/run1-pipeline-v1/showcase/SUMMARY.md](../models/previous-runs/run1-pipeline-v1/showcase/SUMMARY.md).
- Test scenarios: hold, follow, crossing, head_on, double_crossing, slalom, turn, gust; seeds [100000, 100019].
- Stored learned score: 0.512906; baseline: 0.961520; learned verdict: False.
- No collisions or other failures: FAIL; 138/160 episodes completed.
- >= 95% of disturbances recovered within 5 s: FAIL; 38/80 events.
- Slot RMSE below 0.35 m when undisturbed: PASS; hold 0.152 m, follow 0.187 m, turn 0.187 m.
- [stage1-seed0](../models/previous-runs/run1-pipeline-v1/stage1-seed0/best.json): best selection 0.996632 at lineage step 401,408; reason `periodic`.
- [stage2-seed0](../models/previous-runs/run1-pipeline-v1/stage2-seed0/best.json): best selection 0.787784 at lineage step 501,760; reason `periodic`.
- [stage3-seed0](../models/previous-runs/run1-pipeline-v1/stage3-seed0/best.json): best selection 0.495698 at lineage step 501,760; reason `start`.

### run2-pipeline-v2

- Evidence: [models/previous-runs/run2-pipeline-v2/showcase/summary.json](../models/previous-runs/run2-pipeline-v2/showcase/summary.json) and [models/previous-runs/run2-pipeline-v2/showcase/SUMMARY.md](../models/previous-runs/run2-pipeline-v2/showcase/SUMMARY.md).
- Test scenarios: hold, follow, crossing, head_on, double_crossing, slalom, turn, gust; seeds [100000, 100019].
- Stored learned score: 0.814676; baseline: 0.961520; learned verdict: False.
- No collisions or other failures: FAIL; 148/160 episodes completed.
- >= 95% of disturbances recovered within 5 s: FAIL; 66/80 events.
- Slot RMSE below 0.35 m when undisturbed: PASS; hold 0.080 m, follow 0.052 m, turn 0.080 m.
- [stage1-seed0](../models/previous-runs/run2-pipeline-v2/stage1-seed0/best.json): best selection 0.996394 at lineage step 901,120; reason `periodic`.
- [stage2-seed0](../models/previous-runs/run2-pipeline-v2/stage2-seed0/best.json): best selection 0.956076 at lineage step 1,501,184; reason `periodic`.
- [stage3-seed0](../models/previous-runs/run2-pipeline-v2/stage3-seed0/best.json): best selection 0.861480 at lineage step 1,701,888; reason `periodic`.

### run3-pipeline-v3

- Evidence: [models/previous-runs/run3-pipeline-v3/showcase/summary.json](../models/previous-runs/run3-pipeline-v3/showcase/summary.json) and [models/previous-runs/run3-pipeline-v3/showcase/SUMMARY.md](../models/previous-runs/run3-pipeline-v3/showcase/SUMMARY.md).
- Test scenarios: hold, follow, crossing, head_on, double_crossing, turn, gust; seeds [100000, 100019].
- Stored learned score: 0.935661; baseline: 0.970067; learned verdict: False.
- No collisions or other failures: FAIL; 138/140 episodes completed.
- >= 95% of disturbances recovered within 5 s: FAIL; 70/80 events.
- Slot RMSE below 0.35 m when undisturbed: PASS; hold 0.023 m, follow 0.027 m, turn 0.045 m.
- [stage1-seed0](../models/previous-runs/run3-pipeline-v3/stage1-seed0/best.json): best selection 0.996571 at lineage step 700,416; reason `periodic`.
- [stage2-seed0](../models/previous-runs/run3-pipeline-v3/stage2-seed0/best.json): best selection 0.967724 at lineage step 700,416; reason `start`.
- [stage3-seed0](../models/previous-runs/run3-pipeline-v3/stage3-seed0/best.json): best selection 0.956897 at lineage step 700,416; reason `start`.

### Current top-level stage folders

These are local snapshot records, not completed final-test qualifications.

- [stage1-seed0](../models/stage1-seed0/config.json): pipeline 4; stage 1; completed marker present; scenarios hold, follow, turn. Best saved score 0.996023 at lineage step 1,000,000 (`periodic`); [selection evidence](../models/stage1-seed0/best.json).

- [stage2-seed0](../models/stage2-seed0/config.json): pipeline 4; stage 2; completed marker absent; scenarios hold, follow, turn, crossing, head_on. Best saved score 0.963885 at lineage step 1,000,000 (`start`); [selection evidence](../models/stage2-seed0/best.json).

Top-level `models/showcase` present at snapshot: false. Archived showcases under previous-runs do not automatically become the active showcase.


## Missing and historical-only context

The older Markdown mentions `runs/experiments/shield.py`, shield logs, R0/R2/R3 imitation experiments, old numbered-test runs, `training_logs.txt`, and earlier absolute checkout locations such as `neww`. Those names are not evidence that the corresponding file exists in this checkout. The inventory below provides actual retained files.

In particular, knowledge.md §9.1 reports a **separate learned-plus-shield** development experiment: predicted-clearance thresholds/horizons 0.1 m/1 s, 0.2 m/1 s and 0.2 m/1.5 s, with teacher overrides reportedly on 0.01%, 0.03% and 0.24% of drone steps and no failures in 140 development episodes. Its described mechanism is to replace a learned command with the teacher command when predicted clearance drops below the chosen threshold. The referenced implementation/log files are absent here. Preserve these as prose-reported historical results, not verified runtime features or fresh test evidence. `ModelController` runs only the network.

The source paper links were found in existing documentation; no new algorithm was inferred merely from a legacy file label. Missing legacy source cannot be recovered by copying modern formulas into old run notes.

## Discrepancies resolved by current code

1. Current curriculum is 1 M/1 M/1 M steps, seven showcase scenarios, width 128, entropy 0, ten selection seeds and teacher margins 0.7/0.3. Older 5 M total, eight scenarios, width 64, entropy .001 and five selection seeds describe earlier versions.
2. Checkpoint format is v4. Inference accepts v1–v4; training resume accepts v4 only. Some old docs say v3 or mention only v1/v2 restrictions.
3. Failure penalties are culprit-only, not charged to all drones. Reward bounds do not prove all-agent incentive safety; see [actual formula and scope](project_context.md#reward).
4. MAPPO-style value normalization is not implemented; the code performs discounted-return standard-deviation **reward scaling**.
5. Every current stage clones, therefore its derived critic warm-up is ten updates. The five-update setting applies to a different resume-without-cloning path.
6. Replay uses the oldest of up to four 2 s-spaced snapshots, not a fixed 2 s pre-collision state.
7. PPO's total value squared-error coefficient is 0.25. Advantages are normalized over the whole rollout. KL is checked after an epoch. Only the final actor weight is explicitly orthogonally initialized.
8. The actor is order invariant, but adding a new neighbor can change its output. Shared parameter shapes do not prove team-size generalization.
9. `recovered_events` in aggregates means any eventual recovery on successful episodes; the acceptance test separately applies the 5 s limit. Stored old reports are not automatically recalculated with the new overlap rule.
10. Default learned inference uses no safety filter. The `hard_margin` in the teacher is still a soft cost; PyBullet/ORCA/MADDPG/AWAC/JSRL/full MPPI are not implemented.
11. Workers use batched neural inference but sequential Python environment steps. Resume is not exact RNG/environment replay.
12. Current viewer supports automatic best-model reload; older demo prose describes a fixed model. `runs/` is not currently ignored by Git. Old `neww` and Windows paths are provenance, not execution instructions for this checkout.
13. The repository contains 100 legacy archived result JSONs, not the 111 count in one historical timeline. Counts of passing tests in old documents are dated observations, not a freshly executed test count.
14. Slalom/dense/terrain and alternate formation geometries remain available, but are not part of the default learned-policy curriculum. Removal of slalom changes benchmark scope and must accompany result comparisons.

## Retained legacy result packs

Each item links to the full JSON with original checkpoint provenance, all gate values and extra metadata. Binary models, figures and trajectories are linked in the [file inventory](file_inventory.md#results).

Snapshot: **100** packs; modes {'baseline': 53, 'train': 38, 'prod': 9}; packs passing all recorded gates: **3**.

- [20260918_204915_baseline](../results/archive/20260918_204915_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_204929_baseline](../results/archive/20260918_204929_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.4478157444503094, hold_failed.
- [20260918_204947_baseline](../results/archive/20260918_204947_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_204955_baseline](../results/archive/20260918_204955_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.4478157444503094, hold_failed.
- [20260918_205058_train](../results/archive/20260918_205058_train/result.json) — train/rl, seed 0; 1: fail, error 10.0, empty.
- [20260918_205225_baseline](../results/archive/20260918_205225_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_205335_train](../results/archive/20260918_205335_train/result.json) — train/rl, seed 0; 1: fail, error 5.182502082719433, collision_or_bounds.
- [20260918_205754_baseline](../results/archive/20260918_205754_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_210539_baseline](../results/archive/20260918_210539_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.4478114496725903, hold_failed.
- [20260918_210735_baseline](../results/archive/20260918_210735_baseline/result.json) — baseline/baseline, seed 0; 2: fail, error 0.463085709229956, short_episode.
- [20260918_210850_train](../results/archive/20260918_210850_train/result.json) — train/rl, seed 0; 1: fail, error 4.010884374673003, collision_or_bounds.
- [20260918_211217_baseline](../results/archive/20260918_211217_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_211250_baseline](../results/archive/20260918_211250_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_211413_train](../results/archive/20260918_211413_train/result.json) — train/rl, seed 0; 1: fail, error 7.069822811355396, collision_or_bounds; 2: fail, error 7.057859927763161, collision_or_bounds; 3: fail, error 8.670669818318158, collision_or_bounds.
- [20260918_211515_baseline](../results/archive/20260918_211515_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_211605_train](../results/archive/20260918_211605_train/result.json) — train/rl, seed 0; 1: fail, error 5.765952159407723, collision_or_bounds; 2: fail, error 6.503295713220659, collision_or_bounds; 3: fail, error 6.01846750840387, collision_or_bounds.
- [20260918_211650_baseline](../results/archive/20260918_211650_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_211805_train](../results/archive/20260918_211805_train/result.json) — train/rl, seed 0; 1: fail, error 6.080035866114181, collision_or_bounds; 2: fail, error 5.657352335203013, collision_or_bounds; 3: fail, error 5.777607059143398, collision_or_bounds.
- [20260918_211855_baseline](../results/archive/20260918_211855_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_211905_baseline](../results/archive/20260918_211905_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_212159_train](../results/archive/20260918_212159_train/result.json) — train/rl, seed 0; 1: fail, error 1.604521995709178, collision_or_bounds; 2: fail, error 1.6488224307318038, hold_failed; 3: fail, error 3.5321858665035144, collision_or_bounds.
- [20260918_212328_baseline](../results/archive/20260918_212328_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_212351_baseline](../results/archive/20260918_212351_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_212518_train](../results/archive/20260918_212518_train/result.json) — train/rl, seed 0; 1: fail, error 0.5645048076507514, collision_or_bounds; 2: fail, error 0.7326868463483218, hold_failed; 3: fail, error 2.0682040755361597, collision_or_bounds.
- [20260918_212840_train](../results/archive/20260918_212840_train/result.json) — train/rl, seed 0; 1: fail, error 5.521752647713686, collision_or_bounds.
- [20260918_213151_prod](../results/archive/20260918_213151_prod/result.json) — prod/rl, seed 0; 1: fail, error 0.5645048076507514, collision_or_bounds; 2: fail, error 0.7326868463483218, hold_failed; 3: fail, error 2.0682040755361597, collision_or_bounds.
- [20260918_213356_train](../results/archive/20260918_213356_train/result.json) — train/rl, seed 0; 1: fail, error 7.3346069129685, collision_or_bounds; 2: fail, error 7.380476971029665, collision_or_bounds; 3: fail, error 7.103377405627276, collision_or_bounds.
- [20260918_220047_baseline](../results/archive/20260918_220047_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_220548_train](../results/archive/20260918_220548_train/result.json) — train/rl, seed 0; 1: fail, error 2.6123722728277485, collision_or_bounds; 2: fail, error 3.039616954972688, collision_or_bounds; 3: fail, error 2.388480802842305, collision_or_bounds.
- [20260918_220650_baseline](../results/archive/20260918_220650_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.4129866052653279, hold_failed; 2: pass, error 0.04496240659654404, ok; 3: fail, error 1.0051422416487932, collision_or_bounds.
- [20260918_220806_baseline](../results/archive/20260918_220806_baseline/result.json) — baseline/baseline, seed 0; 1: pass, error 0.30327336020749385, ok; 2: pass, error 0.04496240659654404, ok; 3: fail, error 0.9346283423218787, collision_or_bounds.
- [20260918_220900_baseline](../results/archive/20260918_220900_baseline/result.json) — baseline/baseline, seed 0; 1: pass, error 0.3512160081419536, ok; 2: pass, error 0.04496240659654404, ok; 3: fail, error 0.9985092512658797, collision_or_bounds.
- [20260918_220941_baseline](../results/archive/20260918_220941_baseline/result.json) — baseline/baseline, seed 0; 1: pass, error 0.3512160081419536, ok; 2: pass, error 0.04496240659654404, ok; 3: fail, error 1.0131876909549729, hold_failed.
- [20260918_221021_baseline](../results/archive/20260918_221021_baseline/result.json) — baseline/baseline, seed 0; 1: pass, error 0.3512160081419536, ok; 2: pass, error 0.04496240659654404, ok; 3: fail, error 4.006112610786357, collision_or_bounds.
- [20260918_221307_train](../results/archive/20260918_221307_train/result.json) — train/rl, seed 0; 1: fail, error 0.3347242988450478, hold_failed; 2: fail, error 0.11247848140990722, hold_failed; 3: fail, error 0.8647658703552403, hold_failed.
- [20260918_221343_prod](../results/archive/20260918_221343_prod/result.json) — prod/rl, seed 0; 1: fail, error 0.3347242988450478, hold_failed; 2: fail, error 0.11247848140990722, hold_failed; 3: fail, error 0.8647658703552403, hold_failed.
- [20260918_221511_prod](../results/archive/20260918_221511_prod/result.json) — prod/rl, seed 0; 1: fail, error 0.48338802436245265, short_episode.
- [20260918_221811_prod](../results/archive/20260918_221811_prod/result.json) — prod/rl, seed 0; 2: fail, error 0.5921627942178755, hold_failed.
- [20260918_222731_baseline](../results/archive/20260918_222731_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_223735_baseline](../results/archive/20260918_223735_baseline/result.json) — baseline/baseline, seed 0; 2: fail, error 1.6793829678728927, collision_or_bounds.
- [20260918_224027_baseline](../results/archive/20260918_224027_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_224618_baseline](../results/archive/20260918_224618_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_224628_baseline](../results/archive/20260918_224628_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_225350_baseline](../results/archive/20260918_225350_baseline/result.json) — baseline/baseline, seed 0; 2: fail, error 0.3393461383553578, short_episode.
- [20260918_225449_baseline](../results/archive/20260918_225449_baseline/result.json) — baseline/baseline, seed 0; 3: fail, error 0.9723379825983615, short_episode.
- [20260918_225635_baseline](../results/archive/20260918_225635_baseline/result.json) — baseline/baseline, seed 0; 3: fail, error 0.7492661340258451, short_episode.
- [20260918_230216_train](../results/archive/20260918_230216_train/result.json) — train/rl, seed 0; 1: fail, error 0.33032784659507014, collision_or_bounds; 2: fail, error 0.5576213273725786, hold_failed; 3: fail, error 0.8528443478200318, collision_or_bounds.
- [20260918_230341_train](../results/archive/20260918_230341_train/result.json) — train/rl, seed 0; 2: fail, error 3.4191572257939002, collision_or_bounds.
- [20260918_230604_train](../results/archive/20260918_230604_train/result.json) — train/rl, seed 0; 1: fail, error 5.205426900391853, collision_or_bounds.
- [20260918_231033_train](../results/archive/20260918_231033_train/result.json) — train/rl, seed 0; 1: fail, error 7.190841138617707, collision_or_bounds; 2: fail, error 7.223924047990171, collision_or_bounds; 3: fail, error 5.798905235745586, collision_or_bounds.
- [20260918_232803_train](../results/archive/20260918_232803_train/result.json) — train/rl, seed 0; 1: fail, error 7.050821810602568, collision_or_bounds; 2: fail, error 5.624584601705827, collision_or_bounds; 3: fail, error 7.455886924262317, collision_or_bounds.
- [20260918_233620_baseline](../results/archive/20260918_233620_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_233638_baseline](../results/archive/20260918_233638_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_235124_baseline](../results/archive/20260918_235124_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260918_235206_train](../results/archive/20260918_235206_train/result.json) — train/rl, seed 0; 1: fail, error 0.33032784659507014, collision_or_bounds; 2: fail, error 0.5576213273725786, hold_failed; 3: fail, error 0.8528443478200318, collision_or_bounds.
- [20260919_000357_train](../results/archive/20260919_000357_train/result.json) — train/rl, seed 0; 1: fail, error 0.05917108263677807, collision_or_bounds; 2: fail, error 0.5576213273725786, hold_failed; 3: fail, error 0.5529606756919193, collision_or_bounds.
- [20260919_000405_train](../results/archive/20260919_000405_train/result.json) — train/rl, seed 0; 2: fail, error 0.6741104319165087, short_episode.
- [20260919_000539_train](../results/archive/20260919_000539_train/result.json) — train/rl, seed 0; 2: fail, error 4.031813957297131, collision_or_bounds.
- [20260919_000816_prod](../results/archive/20260919_000816_prod/result.json) — prod/rl, seed 0; 2: fail, error 0.7289483582949743, collision_or_bounds.
- [20260919_001249_baseline](../results/archive/20260919_001249_baseline/result.json) — baseline/baseline, seed 0; 2: fail, error 0.6163186935019238, short_episode.
- [20260919_001458_baseline](../results/archive/20260919_001458_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260919_002633_train](../results/archive/20260919_002633_train/result.json) — train/rl, seed 0; 1: fail, error 0.7961493136919714, collision_or_bounds; 2: fail, error 0.80710347835315, hold_failed; 3: fail, error 1.3126793673631163, collision_or_bounds.
- [20260919_002646_train](../results/archive/20260919_002646_train/result.json) — train/rl, seed 0; 2: fail, error 0.9488674511206909, short_episode.
- [20260919_003112_prod](../results/archive/20260919_003112_prod/result.json) — prod/rl, seed 0; 2: fail, error 0.9838614076181098, collision_or_bounds.
- [20260919_004509_train](../results/archive/20260919_004509_train/result.json) — train/rl, seed 0; 1: fail, error 0.7961493136919714, collision_or_bounds; 2: fail, error 0.80710347835315, hold_failed; 3: fail, error 1.3126793673631163, collision_or_bounds.
- [20260919_004642_train](../results/archive/20260919_004642_train/result.json) — train/rl, seed 0; 2: fail, error 3.7968383854149566, collision_or_bounds.
- [20260919_004820_prod](../results/archive/20260919_004820_prod/result.json) — prod/rl, seed 0; 2: fail, error 0.40654695523258183, short_episode.
- [20260919_010338_train](../results/archive/20260919_010338_train/result.json) — train/rl, seed 0; 1: fail, error 0.9009168395255228, collision_or_bounds; 2: fail, error 0.7703554016193563, hold_failed; 3: fail, error 1.40457924939436, collision_or_bounds.
- [20260919_010345_train](../results/archive/20260919_010345_train/result.json) — train/rl, seed 0; 2: fail, error 0.6819507552206912, short_episode.
- [20260919_010413_train](../results/archive/20260919_010413_train/result.json) — train/rl, seed 0; 2: fail, error 1.164802667356004, short_episode.
- [20260919_011540_prod](../results/archive/20260919_011540_prod/result.json) — prod/rl, seed 0; 2: fail, error 0.9617477012392391, short_episode.
- [20260919_011941_baseline](../results/archive/20260919_011941_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260919_012317_baseline](../results/archive/20260919_012317_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260919_012604_baseline](../results/archive/20260919_012604_baseline/result.json) — baseline/baseline, seed 0; 2: fail, error 103.91407390875159, short_episode.
- [20260919_013434_baseline](../results/archive/20260919_013434_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260919_013453_baseline](../results/archive/20260919_013453_baseline/result.json) — baseline/baseline, seed 0; 1: pass, error 17.49302819967417, ok; 2: pass, error 2.2481373610449906, ok; 3: pass, error 22.4181666099075, ok.
- [20260919_013921_train](../results/archive/20260919_013921_train/result.json) — train/rl, seed 0; 1: pass, error 5.857730298283345, ok; 2: fail, error 194.1945081011921, collision_or_bounds; 3: fail, error 100.49462477430875, collision_or_bounds.
- [20260919_014059_baseline](../results/archive/20260919_014059_baseline/result.json) — baseline/baseline, seed 0; 2: fail, error 46.33880417171876, short_episode.
- [20260919_014115_baseline](../results/archive/20260919_014115_baseline/result.json) — baseline/baseline, seed 0; 1: pass, error 17.49302819967417, ok; 2: pass, error 2.2481373610449906, ok; 3: pass, error 22.4181666099075, ok.
- [20260919_014659_train](../results/archive/20260919_014659_train/result.json) — train/rl, seed 0; 1: pass, error 5.857730298283345, ok; 2: fail, error 194.1945081011921, collision_or_bounds; 3: fail, error 100.49462477430875, collision_or_bounds.
- [20260919_014834_train](../results/archive/20260919_014834_train/result.json) — train/rl, seed 0; 2: fail, error 91.6490209277248, short_episode.
- [20260919_015102_prod](../results/archive/20260919_015102_prod/result.json) — prod/rl, seed 0; 2: fail, error 84.61535156679419, hold_failed.
- [20260919_064626_baseline](../results/archive/20260919_064626_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 521.2402939222138, short_episode.
- [20260919_064719_baseline](../results/archive/20260919_064719_baseline/result.json) — baseline/baseline, seed 0; 2: fail, error 30.712656391576708, short_episode.
- [20260919_064742_baseline](../results/archive/20260919_064742_baseline/result.json) — baseline/baseline, seed 0; 3: fail, error 24.876461799243422, short_episode.
- [20260919_065328_baseline](../results/archive/20260919_065328_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260919_065336_baseline](../results/archive/20260919_065336_baseline/result.json) — baseline/baseline, seed 0; 1: pass, error 17.493054853179665, ok; 2: pass, error 1.304180971676753, ok; 3: pass, error 25.62059722478112, ok.
- [20260919_065420_baseline](../results/archive/20260919_065420_baseline/result.json) — baseline/baseline, seed 0; 2: fail, error 15.976902477456608, short_episode.
- [20260919_065842_baseline](../results/archive/20260919_065842_baseline/result.json) — baseline/baseline, seed 0; 2: fail, error 45.543844043654516, short_episode.
- [20260919_070109_baseline](../results/archive/20260919_070109_baseline/result.json) — baseline/baseline, seed 0; 3: fail, error 30.307139904982872, hold_failed.
- [20260919_070311_baseline](../results/archive/20260919_070311_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 18.532854714913302, hold_failed.
- [20260921_022123_baseline](../results/archive/20260921_022123_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260921_022818_train](../results/archive/20260921_022818_train/result.json) — train/rl, seed 0; 1: fail, error 386.95349850908644, collision_or_bounds; 2: fail, error 457.11681562405965, collision_or_bounds; 3: fail, error 498.851675153801, collision_or_bounds.
- [20260921_023101_train](../results/archive/20260921_023101_train/result.json) — train/rl, seed 0; 1: fail, error 61.252095991716914, collision_or_bounds; 2: fail, error 40.08103960094485, hold_failed; 3: fail, error 84.70159561770919, collision_or_bounds.
- [20260921_023327_train](../results/archive/20260921_023327_train/result.json) — train/rl, seed 0; 1: fail, error 62.02518748272907, hold_failed; 2: fail, error 54.279352067547414, hold_failed; 3: fail, error 284.0857144365852, collision_or_bounds.
- [20260921_024056_train](../results/archive/20260921_024056_train/result.json) — train/rl, seed 0; 1: fail, error 559.3650736001714, collision_or_bounds; 2: fail, error 414.9810919009134, collision_or_bounds; 3: fail, error 598.3529280377181, collision_or_bounds.
- [20260921_024558_baseline](../results/archive/20260921_024558_baseline/result.json) — baseline/baseline, seed 0; 1: fail, error 0.5, ok.
- [20260921_024637_train](../results/archive/20260921_024637_train/result.json) — train/rl, seed 0; 1: fail, error 422.41572380404904, collision_or_bounds; 2: fail, error 459.8626395147479, collision_or_bounds; 3: fail, error 519.1367626639108, collision_or_bounds.
- [20260921_025137_train](../results/archive/20260921_025137_train/result.json) — train/rl, seed 0; 1: fail, error 368.82596399137685, collision_or_bounds; 2: fail, error 356.69195964975165, collision_or_bounds; 3: fail, error 362.1558387106529, collision_or_bounds.
- [20260921_025852_train](../results/archive/20260921_025852_train/result.json) — train/rl, seed 0; 1: fail, error 45.20593829734949, hold_failed; 2: fail, error 172.24062638830387, hold_failed; 3: fail, error 144.64177397707786, hold_failed.
