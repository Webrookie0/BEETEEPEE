# Complete project file inventory

[Implementation sequence](README.md) · [Paper index](README.md#paper-index) · [Historical evidence](history.md)

This index covers every retained project file found in the checkout on 29 September 2026, including local untracked stage artifacts. It links large/raw artifacts instead of duplicating their bytes. Equations, configuration, implementation concepts and source references are consolidated in this folder; complete numeric trajectories, logs, images and weights remain in their original files.

Generated virtual environments, dependency caches, pytest caches, Python bytecode, egg-info, .DS_Store and Git internals are excluded from the per-file index. Their roles are described below. The context folder itself is indexed by its README. No source code for components that appear only as legacy graph labels is assumed to exist.

## Source and entry points

### src/swarmflow/__init__.py

[src/swarmflow/__init__.py](../src/swarmflow/__init__.py) — Package exports and version; exposes Config/SwarmEnv.



### src/swarmflow/__main__.py

[src/swarmflow/__main__.py](../src/swarmflow/__main__.py) — Module entry point calling CLI main.



### src/swarmflow/baseline.py

[src/swarmflow/baseline.py](../src/swarmflow/baseline.py) — Sampled predictive controller, optional temperature blend and slot-only ablation. Read Papers 6/28/29.

- [`PredictiveController`](../src/swarmflow/baseline.py#L7)
- [`PredictiveController.__init__`](../src/swarmflow/baseline.py#L10)
- [`PredictiveController.action`](../src/swarmflow/baseline.py#L19)
- [`PredictiveController.__call__`](../src/swarmflow/baseline.py#L59)
- [`SlotController`](../src/swarmflow/baseline.py#L63)
- [`SlotController.__init__`](../src/swarmflow/baseline.py#L66)
- [`SlotController.__call__`](../src/swarmflow/baseline.py#L69)

### src/swarmflow/cli.py

[src/swarmflow/cli.py](../src/swarmflow/cli.py) — Command parser and dispatch; command-specific environment overrides.

- [`main`](../src/swarmflow/cli.py#L16)
- [`train_command`](../src/swarmflow/cli.py#L196)

### src/swarmflow/config.py

[src/swarmflow/config.py](../src/swarmflow/config.py) — Validated simulation defaults and geometry/world names. Read configuration.md.

- [`Config`](../src/swarmflow/config.py#L9)
- [`Config.__post_init__`](../src/swarmflow/config.py#L35)
- [`Config.to_dict`](../src/swarmflow/config.py#L67)

### src/swarmflow/curriculum.py

[src/swarmflow/curriculum.py](../src/swarmflow/curriculum.py) — Stages, versioning/archiving, resume, best model, showcase, training/viewer reports. Read Papers 22–26/30.

- [`Stage`](../src/swarmflow/curriculum.py#L20)
- [`stage_dir`](../src/swarmflow/curriculum.py#L58)
- [`stage_arguments`](../src/swarmflow/curriculum.py#L62)
- [`train_stage`](../src/swarmflow/curriculum.py#L75)
- [`_pipeline_version`](../src/swarmflow/curriculum.py#L101)
- [`_archive`](../src/swarmflow/curriculum.py#L112)
- [`_archive_outdated`](../src/swarmflow/curriculum.py#L119)
- [`run_pipeline`](../src/swarmflow/curriculum.py#L127)
- [`trained_runs`](../src/swarmflow/curriculum.py#L147)
- [`training_chain`](../src/swarmflow/curriculum.py#L159)
- [`_stage_label`](../src/swarmflow/curriculum.py#L170)
- [`build_showcase`](../src/swarmflow/curriculum.py#L178)
- [`_stage_record`](../src/swarmflow/curriculum.py#L260)
- [`_stage_record.read`](../src/swarmflow/curriculum.py#L261)
- [`_summary_markdown`](../src/swarmflow/curriculum.py#L274)
- [`_summary_markdown.cell`](../src/swarmflow/curriculum.py#L275)
- [`default_model`](../src/swarmflow/curriculum.py#L307)
- [`_read_jsonl`](../src/swarmflow/curriculum.py#L319)
- [`_thin`](../src/swarmflow/curriculum.py#L331)
- [`_history_folders`](../src/swarmflow/curriculum.py#L338)
- [`model_report`](../src/swarmflow/curriculum.py#L347)
- [`training_status`](../src/swarmflow/curriculum.py#L391)
- [`training_status.touched`](../src/swarmflow/curriculum.py#L398)
- [`start_viewer`](../src/swarmflow/curriculum.py#L436)
- [`start_viewer.listening`](../src/swarmflow/curriculum.py#L444)

### src/swarmflow/env.py

[src/swarmflow/env.py](../src/swarmflow/env.py) — ParallelEnv, observations, integration, collision, reward, recovery, metrics and state capture. Read Papers 1/2/5/12 and project_context.md.

- [`SwarmEnv`](../src/swarmflow/env.py#L19)
- [`SwarmEnv.__init__`](../src/swarmflow/env.py#L22)
- [`SwarmEnv.observation_space`](../src/swarmflow/env.py#L41)
- [`SwarmEnv.action_space`](../src/swarmflow/env.py#L48)
- [`SwarmEnv._terrain_world`](../src/swarmflow/env.py#L51)
- [`SwarmEnv._min_fly_z`](../src/swarmflow/env.py#L54)
- [`SwarmEnv.target`](../src/swarmflow/env.py#L58)
- [`SwarmEnv.slots`](../src/swarmflow/env.py#L64)
- [`SwarmEnv.reset`](../src/swarmflow/env.py#L82)
- [`SwarmEnv._maybe_stream_world`](../src/swarmflow/env.py#L114)
- [`SwarmEnv.active_obstacles`](../src/swarmflow/env.py#L151)
- [`SwarmEnv.inject`](../src/swarmflow/env.py#L155)
- [`SwarmEnv.steer`](../src/swarmflow/env.py#L169)
- [`SwarmEnv._object_rows`](../src/swarmflow/env.py#L189)
- [`SwarmEnv._ray_hit`](../src/swarmflow/env.py#L205)
- [`SwarmEnv._fill_ray_rows`](../src/swarmflow/env.py#L220)
- [`SwarmEnv._observe`](../src/swarmflow/env.py#L240)
- [`SwarmEnv._clearance`](../src/swarmflow/env.py#L261)
- [`SwarmEnv._update_recovery`](../src/swarmflow/env.py#L280)
- [`SwarmEnv.step`](../src/swarmflow/env.py#L307)
- [`SwarmEnv._reward`](../src/swarmflow/env.py#L392)
- [`SwarmEnv.summary`](../src/swarmflow/env.py#L411)
- [`SwarmEnv.snapshot`](../src/swarmflow/env.py#L423)
- [`SwarmEnv.capture_state`](../src/swarmflow/env.py#L436)
- [`SwarmEnv.restore_state`](../src/swarmflow/env.py#L448)
- [`SwarmEnv.close`](../src/swarmflow/env.py#L459)

### src/swarmflow/evaluate.py

[src/swarmflow/evaluate.py](../src/swarmflow/evaluate.py) — Episode rollout, aggregation, selection score, learned checks and report comparison. Read Papers 24–26.

- [`run_episode`](../src/swarmflow/evaluate.py#L17)
- [`_success_rate`](../src/swarmflow/evaluate.py#L30)
- [`aggregate`](../src/swarmflow/evaluate.py#L37)
- [`selection_score`](../src/swarmflow/evaluate.py#L66)
- [`learning_checks`](../src/swarmflow/evaluate.py#L79)
- [`compare_reports`](../src/swarmflow/evaluate.py#L101)
- [`evaluate`](../src/swarmflow/evaluate.py#L131)

### src/swarmflow/geometry.py

[src/swarmflow/geometry.py](../src/swarmflow/geometry.py) — Norm limiting and swept closest-point geometry. Read Paper 6 and project collision formulas.

- [`limit_norm`](../src/swarmflow/geometry.py#L4)
- [`swept_distance`](../src/swarmflow/geometry.py#L8)

### src/swarmflow/policy.py

[src/swarmflow/policy.py](../src/swarmflow/policy.py) — Set encoders, actor/critic, distributions and checkpoint inference. Read Papers 1/4/7/9/10/18/19/20.

- [`stack_observations`](../src/swarmflow/policy.py#L17)
- [`stack_teams`](../src/swarmflow/policy.py#L21)
- [`tensors`](../src/swarmflow/policy.py#L26)
- [`SetEncoder`](../src/swarmflow/policy.py#L30)
- [`SetEncoder.__init__`](../src/swarmflow/policy.py#L38)
- [`SetEncoder.forward`](../src/swarmflow/policy.py#L46)
- [`Encoder`](../src/swarmflow/policy.py#L68)
- [`Encoder.__init__`](../src/swarmflow/policy.py#L71)
- [`Encoder.encode`](../src/swarmflow/policy.py#L78)
- [`ActorCritic`](../src/swarmflow/policy.py#L86)
- [`ActorCritic.__init__`](../src/swarmflow/policy.py#L90)
- [`ActorCritic.network_spec`](../src/swarmflow/policy.py#L107)
- [`ActorCritic.act_from_hidden`](../src/swarmflow/policy.py#L110)
- [`ActorCritic.value_local`](../src/swarmflow/policy.py#L113)
- [`ActorCritic.value_central`](../src/swarmflow/policy.py#L116)
- [`ActorCritic._value`](../src/swarmflow/policy.py#L120)
- [`ActorCritic._critic_hidden`](../src/swarmflow/policy.py#L125)
- [`ActorCritic.forward`](../src/swarmflow/policy.py#L128)
- [`ActorCritic.act`](../src/swarmflow/policy.py#L132)
- [`ActorCritic.value`](../src/swarmflow/policy.py#L136)
- [`ActorCritic.critic_parameters`](../src/swarmflow/policy.py#L140)
- [`ActorCritic.actor_parameters`](../src/swarmflow/policy.py#L148)
- [`ActorCritic.distribution`](../src/swarmflow/policy.py#L152)
- [`ActorCritic.log_probability`](../src/swarmflow/policy.py#L158)
- [`ActorCritic.sample`](../src/swarmflow/policy.py#L162)
- [`ActorCritic.evaluate`](../src/swarmflow/policy.py#L167)
- [`ModelController`](../src/swarmflow/policy.py#L172)
- [`ModelController.__init__`](../src/swarmflow/policy.py#L175)
- [`ModelController.__call__`](../src/swarmflow/policy.py#L179)
- [`LearnedController`](../src/swarmflow/policy.py#L185)
- [`LearnedController.__init__`](../src/swarmflow/policy.py#L186)
- [`LearnedController.__call__`](../src/swarmflow/policy.py#L203)

### src/swarmflow/scenarios.py

[src/swarmflow/scenarios.py](../src/swarmflow/scenarios.py) — Formation offsets, object trajectories, scripted target motion and terrain. Read project_context.md.

- [`Obstacle`](../src/swarmflow/scenarios.py#L11)
- [`Obstacle.position`](../src/swarmflow/scenarios.py#L20)
- [`_enforce_min_spacing`](../src/swarmflow/scenarios.py#L24)
- [`_polygon_ring`](../src/swarmflow/scenarios.py#L36)
- [`formation`](../src/swarmflow/scenarios.py#L57)
- [`ground_height`](../src/swarmflow/scenarios.py#L74)
- [`make_peaks`](../src/swarmflow/scenarios.py#L90)
- [`canyon_center_y`](../src/swarmflow/scenarios.py#L109)
- [`make_canyon_segment`](../src/swarmflow/scenarios.py#L114)
- [`make_terrain`](../src/swarmflow/scenarios.py#L127)
- [`target_state`](../src/swarmflow/scenarios.py#L141)
- [`crossing`](../src/swarmflow/scenarios.py#L156)
- [`make_obstacles`](../src/swarmflow/scenarios.py#L171)

### src/swarmflow/server.py

[src/swarmflow/server.py](../src/swarmflow/server.py) — Local HTTP API, controller choice, state lock and best-model follow thread.

- [`_load`](../src/swarmflow/server.py#L21)
- [`_stamp`](../src/swarmflow/server.py#L28)
- [`make_server`](../src/swarmflow/server.py#L32)
- [`make_server.controller_for`](../src/swarmflow/server.py#L40)
- [`make_server.follow_best`](../src/swarmflow/server.py#L52)
- [`make_server.Handler`](../src/swarmflow/server.py#L74)
- [`make_server.Handler.log_message`](../src/swarmflow/server.py#L75)
- [`make_server.Handler.respond`](../src/swarmflow/server.py#L78)
- [`make_server.Handler.snapshot`](../src/swarmflow/server.py#L88)
- [`make_server.Handler.do_GET`](../src/swarmflow/server.py#L93)
- [`make_server.Handler.do_POST`](../src/swarmflow/server.py#L126)
- [`serve`](../src/swarmflow/server.py#L220)

### src/swarmflow/suite.py

[src/swarmflow/suite.py](../src/swarmflow/suite.py) — Eleven numbered reproducible tests, result folders and pass/fail criteria.

- [`Options`](../src/swarmflow/suite.py#L25)
- [`Result`](../src/swarmflow/suite.py#L33)
- [`Test`](../src/swarmflow/suite.py#L40)
- [`_seeds`](../src/swarmflow/suite.py#L54)
- [`_evaluate`](../src/swarmflow/suite.py#L58)
- [`_acceptance`](../src/swarmflow/suite.py#L68)
- [`_scenario_table`](../src/swarmflow/suite.py#L99)
- [`_baseline_test`](../src/swarmflow/suite.py#L110)
- [`_baseline_test.run`](../src/swarmflow/suite.py#L111)
- [`run_unit_checks`](../src/swarmflow/suite.py#L120)
- [`run_no_avoidance`](../src/swarmflow/suite.py#L131)
- [`_train_or_reuse`](../src/swarmflow/suite.py#L141)
- [`run_smoke_training`](../src/swarmflow/suite.py#L155)
- [`_learned_vs_baseline`](../src/swarmflow/suite.py#L167)
- [`_run_stage`](../src/swarmflow/suite.py#L184)
- [`_run_stage.run`](../src/swarmflow/suite.py#L185)
- [`_folder`](../src/swarmflow/suite.py#L275)
- [`catalog`](../src/swarmflow/suite.py#L280)
- [`print_catalog`](../src/swarmflow/suite.py#L296)
- [`resolve`](../src/swarmflow/suite.py#L306)
- [`prompt`](../src/swarmflow/suite.py#L323)
- [`run_tests`](../src/swarmflow/suite.py#L332)

### src/swarmflow/train.py

[src/swarmflow/train.py](../src/swarmflow/train.py) — Teacher data, DAgger, reward scaling, GAE, PPO, replay workers, validation and saves. Read Papers 2/7–17/27/30.

- [`advantages`](../src/swarmflow/train.py#L26)
- [`RewardScaler`](../src/swarmflow/train.py#L37)
- [`RewardScaler.__init__`](../src/swarmflow/train.py#L40)
- [`RewardScaler.__call__`](../src/swarmflow/train.py#L45)
- [`RewardScaler._update`](../src/swarmflow/train.py#L53)
- [`RewardScaler.state`](../src/swarmflow/train.py#L60)
- [`Worker`](../src/swarmflow/train.py#L64)
- [`Worker.__init__`](../src/swarmflow/train.py#L67)
- [`Worker.start_episode`](../src/swarmflow/train.py#L74)
- [`Worker.step`](../src/swarmflow/train.py#L80)
- [`_hard_share`](../src/swarmflow/train.py#L107)
- [`_demo_tensors`](../src/swarmflow/train.py#L117)
- [`_demo_ids`](../src/swarmflow/train.py#L129)
- [`_fit_clone`](../src/swarmflow/train.py#L137)
- [`_collect`](../src/swarmflow/train.py#L155)
- [`clone`](../src/swarmflow/train.py#L184)
- [`demonstrations`](../src/swarmflow/train.py#L214)
- [`_clock`](../src/swarmflow/train.py#L223)
- [`train`](../src/swarmflow/train.py#L228)
- [`train.write`](../src/swarmflow/train.py#L349)
- [`train.save`](../src/swarmflow/train.py#L353)
- [`train.validate`](../src/swarmflow/train.py#L366)

### src/swarmflow/web/app.js

[src/swarmflow/web/app.js](../src/swarmflow/web/app.js) — Canvas rendering, pilot controls, API client, telemetry, training curves, tests and PNG export.

- [`notify`](../src/swarmflow/web/app.js#L14)
- [`api`](../src/swarmflow/web/app.js#L15)
- [`liveError`](../src/swarmflow/web/app.js#L29)
- [`accept`](../src/swarmflow/web/app.js#L33)
- [`syncController`](../src/swarmflow/web/app.js#L78)
- [`steerVelocity`](../src/swarmflow/web/app.js#L89)
- [`steer`](../src/swarmflow/web/app.js#L100)
- [`reset`](../src/swarmflow/web/app.js#L106)
- [`inject`](../src/swarmflow/web/app.js#L158)
- [`setView`](../src/swarmflow/web/app.js#L180)
- [`fit`](../src/swarmflow/web/app.js#L204)
- [`project`](../src/swarmflow/web/app.js#L211)
- [`line`](../src/swarmflow/web/app.js#L223)
- [`circle`](../src/swarmflow/web/app.js#L224)
- [`drawDrone`](../src/swarmflow/web/app.js#L225)
- [`drawObject`](../src/swarmflow/web/app.js#L237)
- [`groundAt`](../src/swarmflow/web/app.js#L244)
- [`drawScene`](../src/swarmflow/web/app.js#L253)
- [`draw`](../src/swarmflow/web/app.js#L293)
- [`drawChart`](../src/swarmflow/web/app.js#L319)
- [`tick`](../src/swarmflow/web/app.js#L328)
- [`scenarioLabel`](../src/swarmflow/web/app.js#L343)
- [`el`](../src/swarmflow/web/app.js#L347)
- [`scenarioHint`](../src/swarmflow/web/app.js#L353)
- [`watch`](../src/swarmflow/web/app.js#L361)
- [`renderTests`](../src/swarmflow/web/app.js#L382)
- [`loadTests`](../src/swarmflow/web/app.js#L418)
- [`loadModel`](../src/swarmflow/web/app.js#L429)
- [`renderTraining`](../src/swarmflow/web/app.js#L436)
- [`renderModel`](../src/swarmflow/web/app.js#L454)
- [`renderModelTable`](../src/swarmflow/web/app.js#L499)
- [`plotSeries`](../src/swarmflow/web/app.js#L515)
- [`drawCurves`](../src/swarmflow/web/app.js#L547)

### src/swarmflow/web/index.html

[src/swarmflow/web/index.html](../src/swarmflow/web/index.html) — Viewer controls, scenario options, plots, model verdict and test panels.

### src/swarmflow/web/style.css

[src/swarmflow/web/style.css](../src/swarmflow/web/style.css) — Viewer styling, responsive layout, colors and typography.

## Tests

These tests exist in source; this documentation task did not retrain models or rerun long benchmark suites. Their presence is not proof of learned-policy qualification.

### test_environment.py

[tests/test_environment.py](../tests/test_environment.py)

- [`test_parallel_api`](../tests/test_environment.py#L14)
- [`test_seed_and_slot_identity`](../tests/test_environment.py#L18)
- [`test_swept_collision_catches_object_between_frames`](../tests/test_environment.py#L37)
- [`test_acceleration_limit_and_action_validation`](../tests/test_environment.py#L49)
- [`test_time_limit_is_truncation`](../tests/test_environment.py#L61)
- [`test_range_mask_and_dropout`](../tests/test_environment.py#L71)
- [`test_recovery_requires_each_original_slot_and_full_hold`](../tests/test_environment.py#L79)
- [`test_recovery_waits_for_a_second_object_inside_the_window`](../tests/test_environment.py#L98)
- [`test_injection_validation_and_persistent_slots`](../tests/test_environment.py#L115)
- [`test_predictive_controller_recovers_but_blind_controller_collides`](../tests/test_environment.py#L126)
- [`test_invalid_config`](../tests/test_environment.py#L138)
- [`test_formation_geometries_centered`](../tests/test_environment.py#L144)
- [`test_pilot_reference_steering`](../tests/test_environment.py#L155)
- [`test_hills_auto_climb_and_hard_contact_still_ends`](../tests/test_environment.py#L168)

### test_learning.py

[tests/test_learning.py](../tests/test_learning.py)

- [`test_gae_bootstraps_truncation_without_crossing_reset`](../tests/test_learning.py#L12)
- [`test_policy_is_invariant_to_object_order_and_ignores_padding`](../tests/test_learning.py#L23)
- [`test_training_checkpoint_reload_and_resume`](../tests/test_learning.py#L46)
- [`test_training_keeps_best_checkpoint_and_validation_log`](../tests/test_learning.py#L70)
- [`test_continue_finishes_the_original_budget`](../tests/test_learning.py#L87)
- [`test_resumed_stage_clones_its_scenarios_and_keeps_an_imitation_loss`](../tests/test_learning.py#L99)
- [`test_continue_rebuilds_teacher_data_for_the_imitation_loss`](../tests/test_learning.py#L117)
- [`test_gap_features_keep_the_encoder_order_invariant_and_round_trip`](../tests/test_learning.py#L129)
- [`test_crash_penalty_goes_to_the_drones_involved`](../tests/test_learning.py#L150)
- [`test_obstacle_warning_starts_one_metre_away`](../tests/test_learning.py#L160)
- [`test_separate_critic_does_not_touch_actor_weights`](../tests/test_learning.py#L170)
- [`test_reward_is_bounded_and_failure_is_always_worse`](../tests/test_learning.py#L181)
- [`test_attention_encoder_permutation_invariance`](../tests/test_learning.py#L192)
- [`test_actor_critic_forward_and_central_value_shapes`](../tests/test_learning.py#L210)
- [`test_restore_state_returns_to_prior_time_and_positions`](../tests/test_learning.py#L233)

### test_showcase.py

[tests/test_showcase.py](../tests/test_showcase.py)

- [`test_showcase_selects_best_run_and_viewer_reports_it`](../tests/test_showcase.py#L22)
- [`test_viewer_follows_new_best_model_during_training`](../tests/test_showcase.py#L66)

## Root documents and build files



- [.gitignore](../.gitignore) — Generated-tool exclusions; runs exclusion is commented out.
- [Launch SwarmFlow.command](../Launch%20SwarmFlow.command) — macOS launcher: environment setup, local server reuse/launch and log path.
- [Makefile](../Makefile) — Setup, training, evaluation, test and viewer command shortcuts.
- [README.md](../README.md) — Top-level setup, feature summary and links; contains some historical wording.
- [guide.md](../guide.md) — Training/continuation/showcase workflow and troubleshooting.
- [knowledge.md](../knowledge.md) — Consolidated history, architecture, experiments, missing legacy-context references and file map.
- [pyproject.toml](../pyproject.toml) — Package metadata, direct dependency constraints, console script and packaged web assets.
- [sofar.md](../sofar.md) — Research conclusions, original six paper-to-feature mappings and earlier status.
- [training.md](../training.md) — Detailed learning explanation, paper reading list, versions and rejected experiments.
- [uv.lock](../uv.lock) — Exact resolved packages, transitive requirements and platform markers.

## Docs

Research plan, old contracts, validation evidence and viewer images.

- [docs/flight-preview.png](../docs/flight-preview.png) — Saved browser demonstration image, not a quantitative research result.
- [docs/implementation.md](../docs/implementation.md) — Earlier implementation contract; reconcile v3 and other stale statements with current code.
- [docs/local-demo-check.png](../docs/local-demo-check.png) — Saved local-viewer verification image.
- [docs/research-plan.md](../docs/research-plan.md) — Original source list, hypotheses, contracts, proposed acceptance and future experiments.
- [docs/training-demo.md](../docs/training-demo.md) — Historical demo commands; its update note and newer guide supersede old launcher/reload claims.
- [docs/validation.md](../docs/validation.md) — Dated small baseline, scaling, ablation and smoke evidence with caveats.

## Models

Active local stages, archived complete pipelines, showcase reports and legacy root checkpoints.


### models

- [models/best.pt](../models/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.
- [models/latest.pt](../models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### models/previous-runs

- [models/previous-runs/README.md](../models/previous-runs/README.md) — Index and diagnosis of three archived full pipelines.

### models/previous-runs/run1-pipeline-v1/showcase

- [models/previous-runs/run1-pipeline-v1/showcase/SUMMARY.md](../models/previous-runs/run1-pipeline-v1/showcase/SUMMARY.md) — Saved showcase selection, training lineage, held-out verdict and per-scenario comparison.
- [models/previous-runs/run1-pipeline-v1/showcase/best.pt](../models/previous-runs/run1-pipeline-v1/showcase/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### models/previous-runs/run1-pipeline-v1/showcase/history/stage1-seed0

- [models/previous-runs/run1-pipeline-v1/showcase/history/stage1-seed0/best.json](../models/previous-runs/run1-pipeline-v1/showcase/history/stage1-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/previous-runs/run1-pipeline-v1/showcase/history/stage1-seed0/config.json](../models/previous-runs/run1-pipeline-v1/showcase/history/stage1-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/previous-runs/run1-pipeline-v1/showcase/history/stage1-seed0/evaluations.jsonl](../models/previous-runs/run1-pipeline-v1/showcase/history/stage1-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/previous-runs/run1-pipeline-v1/showcase/history/stage1-seed0/finished.json](../models/previous-runs/run1-pipeline-v1/showcase/history/stage1-seed0/finished.json) — Completed-run budget, best score and elapsed-time marker.
- [models/previous-runs/run1-pipeline-v1/showcase/history/stage1-seed0/training.jsonl](../models/previous-runs/run1-pipeline-v1/showcase/history/stage1-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### models/previous-runs/run1-pipeline-v1/showcase/history/stage2-seed0

- [models/previous-runs/run1-pipeline-v1/showcase/history/stage2-seed0/best.json](../models/previous-runs/run1-pipeline-v1/showcase/history/stage2-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/previous-runs/run1-pipeline-v1/showcase/history/stage2-seed0/config.json](../models/previous-runs/run1-pipeline-v1/showcase/history/stage2-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/previous-runs/run1-pipeline-v1/showcase/history/stage2-seed0/evaluations.jsonl](../models/previous-runs/run1-pipeline-v1/showcase/history/stage2-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/previous-runs/run1-pipeline-v1/showcase/history/stage2-seed0/finished.json](../models/previous-runs/run1-pipeline-v1/showcase/history/stage2-seed0/finished.json) — Completed-run budget, best score and elapsed-time marker.
- [models/previous-runs/run1-pipeline-v1/showcase/history/stage2-seed0/training.jsonl](../models/previous-runs/run1-pipeline-v1/showcase/history/stage2-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### models/previous-runs/run1-pipeline-v1/showcase/history/stage3-seed0

- [models/previous-runs/run1-pipeline-v1/showcase/history/stage3-seed0/best.json](../models/previous-runs/run1-pipeline-v1/showcase/history/stage3-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/previous-runs/run1-pipeline-v1/showcase/history/stage3-seed0/config.json](../models/previous-runs/run1-pipeline-v1/showcase/history/stage3-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/previous-runs/run1-pipeline-v1/showcase/history/stage3-seed0/evaluations.jsonl](../models/previous-runs/run1-pipeline-v1/showcase/history/stage3-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/previous-runs/run1-pipeline-v1/showcase/history/stage3-seed0/training.jsonl](../models/previous-runs/run1-pipeline-v1/showcase/history/stage3-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### models/previous-runs/run1-pipeline-v1/showcase

- [models/previous-runs/run1-pipeline-v1/showcase/summary.json](../models/previous-runs/run1-pipeline-v1/showcase/summary.json) — Saved showcase selection, training lineage, held-out verdict and per-scenario comparison.
- [models/previous-runs/run1-pipeline-v1/showcase/test_baseline.json](../models/previous-runs/run1-pipeline-v1/showcase/test_baseline.json) — Saved showcase learned/baseline/comparison report.
- [models/previous-runs/run1-pipeline-v1/showcase/test_compare.json](../models/previous-runs/run1-pipeline-v1/showcase/test_compare.json) — Saved showcase learned/baseline/comparison report.
- [models/previous-runs/run1-pipeline-v1/showcase/test_learned.json](../models/previous-runs/run1-pipeline-v1/showcase/test_learned.json) — Saved showcase learned/baseline/comparison report.
- [models/previous-runs/run1-pipeline-v1/showcase/validation.json](../models/previous-runs/run1-pipeline-v1/showcase/validation.json) — Historical validation or copied best-selection evidence; inspect its schema/version.

### models/previous-runs/run1-pipeline-v1/stage1-seed0

- [models/previous-runs/run1-pipeline-v1/stage1-seed0/best.json](../models/previous-runs/run1-pipeline-v1/stage1-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/previous-runs/run1-pipeline-v1/stage1-seed0/best.pt](../models/previous-runs/run1-pipeline-v1/stage1-seed0/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.
- [models/previous-runs/run1-pipeline-v1/stage1-seed0/config.json](../models/previous-runs/run1-pipeline-v1/stage1-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/previous-runs/run1-pipeline-v1/stage1-seed0/evaluations.jsonl](../models/previous-runs/run1-pipeline-v1/stage1-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/previous-runs/run1-pipeline-v1/stage1-seed0/finished.json](../models/previous-runs/run1-pipeline-v1/stage1-seed0/finished.json) — Completed-run budget, best score and elapsed-time marker.
- [models/previous-runs/run1-pipeline-v1/stage1-seed0/training.jsonl](../models/previous-runs/run1-pipeline-v1/stage1-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### models/previous-runs/run1-pipeline-v1/stage2-seed0

- [models/previous-runs/run1-pipeline-v1/stage2-seed0/best.json](../models/previous-runs/run1-pipeline-v1/stage2-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/previous-runs/run1-pipeline-v1/stage2-seed0/best.pt](../models/previous-runs/run1-pipeline-v1/stage2-seed0/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.
- [models/previous-runs/run1-pipeline-v1/stage2-seed0/config.json](../models/previous-runs/run1-pipeline-v1/stage2-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/previous-runs/run1-pipeline-v1/stage2-seed0/evaluations.jsonl](../models/previous-runs/run1-pipeline-v1/stage2-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/previous-runs/run1-pipeline-v1/stage2-seed0/finished.json](../models/previous-runs/run1-pipeline-v1/stage2-seed0/finished.json) — Completed-run budget, best score and elapsed-time marker.
- [models/previous-runs/run1-pipeline-v1/stage2-seed0/training.jsonl](../models/previous-runs/run1-pipeline-v1/stage2-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### models/previous-runs/run1-pipeline-v1/stage3-seed0

- [models/previous-runs/run1-pipeline-v1/stage3-seed0/best.json](../models/previous-runs/run1-pipeline-v1/stage3-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/previous-runs/run1-pipeline-v1/stage3-seed0/best.pt](../models/previous-runs/run1-pipeline-v1/stage3-seed0/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.
- [models/previous-runs/run1-pipeline-v1/stage3-seed0/config.json](../models/previous-runs/run1-pipeline-v1/stage3-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/previous-runs/run1-pipeline-v1/stage3-seed0/evaluations.jsonl](../models/previous-runs/run1-pipeline-v1/stage3-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/previous-runs/run1-pipeline-v1/stage3-seed0/training.jsonl](../models/previous-runs/run1-pipeline-v1/stage3-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### models/previous-runs/run1-pipeline-v1

- [models/previous-runs/run1-pipeline-v1/terminal_log_tail.txt](../models/previous-runs/run1-pipeline-v1/terminal_log_tail.txt) — Historical runtime/training text log.

### models/previous-runs/run2-pipeline-v2/showcase

- [models/previous-runs/run2-pipeline-v2/showcase/SUMMARY.md](../models/previous-runs/run2-pipeline-v2/showcase/SUMMARY.md) — Saved showcase selection, training lineage, held-out verdict and per-scenario comparison.
- [models/previous-runs/run2-pipeline-v2/showcase/best.pt](../models/previous-runs/run2-pipeline-v2/showcase/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### models/previous-runs/run2-pipeline-v2/showcase/history/stage1-seed0

- [models/previous-runs/run2-pipeline-v2/showcase/history/stage1-seed0/best.json](../models/previous-runs/run2-pipeline-v2/showcase/history/stage1-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/previous-runs/run2-pipeline-v2/showcase/history/stage1-seed0/config.json](../models/previous-runs/run2-pipeline-v2/showcase/history/stage1-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/previous-runs/run2-pipeline-v2/showcase/history/stage1-seed0/evaluations.jsonl](../models/previous-runs/run2-pipeline-v2/showcase/history/stage1-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/previous-runs/run2-pipeline-v2/showcase/history/stage1-seed0/finished.json](../models/previous-runs/run2-pipeline-v2/showcase/history/stage1-seed0/finished.json) — Completed-run budget, best score and elapsed-time marker.
- [models/previous-runs/run2-pipeline-v2/showcase/history/stage1-seed0/training.jsonl](../models/previous-runs/run2-pipeline-v2/showcase/history/stage1-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### models/previous-runs/run2-pipeline-v2/showcase/history/stage2-seed0

- [models/previous-runs/run2-pipeline-v2/showcase/history/stage2-seed0/best.json](../models/previous-runs/run2-pipeline-v2/showcase/history/stage2-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/previous-runs/run2-pipeline-v2/showcase/history/stage2-seed0/config.json](../models/previous-runs/run2-pipeline-v2/showcase/history/stage2-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/previous-runs/run2-pipeline-v2/showcase/history/stage2-seed0/evaluations.jsonl](../models/previous-runs/run2-pipeline-v2/showcase/history/stage2-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/previous-runs/run2-pipeline-v2/showcase/history/stage2-seed0/finished.json](../models/previous-runs/run2-pipeline-v2/showcase/history/stage2-seed0/finished.json) — Completed-run budget, best score and elapsed-time marker.
- [models/previous-runs/run2-pipeline-v2/showcase/history/stage2-seed0/training.jsonl](../models/previous-runs/run2-pipeline-v2/showcase/history/stage2-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### models/previous-runs/run2-pipeline-v2/showcase/history/stage3-seed0

- [models/previous-runs/run2-pipeline-v2/showcase/history/stage3-seed0/best.json](../models/previous-runs/run2-pipeline-v2/showcase/history/stage3-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/previous-runs/run2-pipeline-v2/showcase/history/stage3-seed0/config.json](../models/previous-runs/run2-pipeline-v2/showcase/history/stage3-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/previous-runs/run2-pipeline-v2/showcase/history/stage3-seed0/evaluations.jsonl](../models/previous-runs/run2-pipeline-v2/showcase/history/stage3-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/previous-runs/run2-pipeline-v2/showcase/history/stage3-seed0/finished.json](../models/previous-runs/run2-pipeline-v2/showcase/history/stage3-seed0/finished.json) — Completed-run budget, best score and elapsed-time marker.
- [models/previous-runs/run2-pipeline-v2/showcase/history/stage3-seed0/training.jsonl](../models/previous-runs/run2-pipeline-v2/showcase/history/stage3-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### models/previous-runs/run2-pipeline-v2/showcase

- [models/previous-runs/run2-pipeline-v2/showcase/summary.json](../models/previous-runs/run2-pipeline-v2/showcase/summary.json) — Saved showcase selection, training lineage, held-out verdict and per-scenario comparison.
- [models/previous-runs/run2-pipeline-v2/showcase/test_baseline.json](../models/previous-runs/run2-pipeline-v2/showcase/test_baseline.json) — Saved showcase learned/baseline/comparison report.
- [models/previous-runs/run2-pipeline-v2/showcase/test_compare.json](../models/previous-runs/run2-pipeline-v2/showcase/test_compare.json) — Saved showcase learned/baseline/comparison report.
- [models/previous-runs/run2-pipeline-v2/showcase/test_learned.json](../models/previous-runs/run2-pipeline-v2/showcase/test_learned.json) — Saved showcase learned/baseline/comparison report.
- [models/previous-runs/run2-pipeline-v2/showcase/validation.json](../models/previous-runs/run2-pipeline-v2/showcase/validation.json) — Historical validation or copied best-selection evidence; inspect its schema/version.

### models/previous-runs/run2-pipeline-v2/stage1-seed0

- [models/previous-runs/run2-pipeline-v2/stage1-seed0/best.json](../models/previous-runs/run2-pipeline-v2/stage1-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/previous-runs/run2-pipeline-v2/stage1-seed0/best.pt](../models/previous-runs/run2-pipeline-v2/stage1-seed0/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.
- [models/previous-runs/run2-pipeline-v2/stage1-seed0/config.json](../models/previous-runs/run2-pipeline-v2/stage1-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/previous-runs/run2-pipeline-v2/stage1-seed0/evaluations.jsonl](../models/previous-runs/run2-pipeline-v2/stage1-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/previous-runs/run2-pipeline-v2/stage1-seed0/finished.json](../models/previous-runs/run2-pipeline-v2/stage1-seed0/finished.json) — Completed-run budget, best score and elapsed-time marker.
- [models/previous-runs/run2-pipeline-v2/stage1-seed0/training.jsonl](../models/previous-runs/run2-pipeline-v2/stage1-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### models/previous-runs/run2-pipeline-v2/stage2-seed0

- [models/previous-runs/run2-pipeline-v2/stage2-seed0/best.json](../models/previous-runs/run2-pipeline-v2/stage2-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/previous-runs/run2-pipeline-v2/stage2-seed0/best.pt](../models/previous-runs/run2-pipeline-v2/stage2-seed0/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.
- [models/previous-runs/run2-pipeline-v2/stage2-seed0/config.json](../models/previous-runs/run2-pipeline-v2/stage2-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/previous-runs/run2-pipeline-v2/stage2-seed0/evaluations.jsonl](../models/previous-runs/run2-pipeline-v2/stage2-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/previous-runs/run2-pipeline-v2/stage2-seed0/finished.json](../models/previous-runs/run2-pipeline-v2/stage2-seed0/finished.json) — Completed-run budget, best score and elapsed-time marker.
- [models/previous-runs/run2-pipeline-v2/stage2-seed0/training.jsonl](../models/previous-runs/run2-pipeline-v2/stage2-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### models/previous-runs/run2-pipeline-v2/stage3-seed0

- [models/previous-runs/run2-pipeline-v2/stage3-seed0/best.json](../models/previous-runs/run2-pipeline-v2/stage3-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/previous-runs/run2-pipeline-v2/stage3-seed0/best.pt](../models/previous-runs/run2-pipeline-v2/stage3-seed0/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.
- [models/previous-runs/run2-pipeline-v2/stage3-seed0/config.json](../models/previous-runs/run2-pipeline-v2/stage3-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/previous-runs/run2-pipeline-v2/stage3-seed0/evaluations.jsonl](../models/previous-runs/run2-pipeline-v2/stage3-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/previous-runs/run2-pipeline-v2/stage3-seed0/finished.json](../models/previous-runs/run2-pipeline-v2/stage3-seed0/finished.json) — Completed-run budget, best score and elapsed-time marker.
- [models/previous-runs/run2-pipeline-v2/stage3-seed0/training.jsonl](../models/previous-runs/run2-pipeline-v2/stage3-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### models/previous-runs/run3-pipeline-v3/showcase

- [models/previous-runs/run3-pipeline-v3/showcase/SUMMARY.md](../models/previous-runs/run3-pipeline-v3/showcase/SUMMARY.md) — Saved showcase selection, training lineage, held-out verdict and per-scenario comparison.
- [models/previous-runs/run3-pipeline-v3/showcase/best.pt](../models/previous-runs/run3-pipeline-v3/showcase/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### models/previous-runs/run3-pipeline-v3/showcase/history/stage1-seed0

- [models/previous-runs/run3-pipeline-v3/showcase/history/stage1-seed0/best.json](../models/previous-runs/run3-pipeline-v3/showcase/history/stage1-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/previous-runs/run3-pipeline-v3/showcase/history/stage1-seed0/config.json](../models/previous-runs/run3-pipeline-v3/showcase/history/stage1-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/previous-runs/run3-pipeline-v3/showcase/history/stage1-seed0/evaluations.jsonl](../models/previous-runs/run3-pipeline-v3/showcase/history/stage1-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/previous-runs/run3-pipeline-v3/showcase/history/stage1-seed0/finished.json](../models/previous-runs/run3-pipeline-v3/showcase/history/stage1-seed0/finished.json) — Completed-run budget, best score and elapsed-time marker.
- [models/previous-runs/run3-pipeline-v3/showcase/history/stage1-seed0/training.jsonl](../models/previous-runs/run3-pipeline-v3/showcase/history/stage1-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### models/previous-runs/run3-pipeline-v3/showcase/history/stage2-seed0

- [models/previous-runs/run3-pipeline-v3/showcase/history/stage2-seed0/best.json](../models/previous-runs/run3-pipeline-v3/showcase/history/stage2-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/previous-runs/run3-pipeline-v3/showcase/history/stage2-seed0/config.json](../models/previous-runs/run3-pipeline-v3/showcase/history/stage2-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/previous-runs/run3-pipeline-v3/showcase/history/stage2-seed0/evaluations.jsonl](../models/previous-runs/run3-pipeline-v3/showcase/history/stage2-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/previous-runs/run3-pipeline-v3/showcase/history/stage2-seed0/finished.json](../models/previous-runs/run3-pipeline-v3/showcase/history/stage2-seed0/finished.json) — Completed-run budget, best score and elapsed-time marker.
- [models/previous-runs/run3-pipeline-v3/showcase/history/stage2-seed0/training.jsonl](../models/previous-runs/run3-pipeline-v3/showcase/history/stage2-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### models/previous-runs/run3-pipeline-v3/showcase/history/stage3-seed0

- [models/previous-runs/run3-pipeline-v3/showcase/history/stage3-seed0/best.json](../models/previous-runs/run3-pipeline-v3/showcase/history/stage3-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/previous-runs/run3-pipeline-v3/showcase/history/stage3-seed0/config.json](../models/previous-runs/run3-pipeline-v3/showcase/history/stage3-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/previous-runs/run3-pipeline-v3/showcase/history/stage3-seed0/evaluations.jsonl](../models/previous-runs/run3-pipeline-v3/showcase/history/stage3-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/previous-runs/run3-pipeline-v3/showcase/history/stage3-seed0/finished.json](../models/previous-runs/run3-pipeline-v3/showcase/history/stage3-seed0/finished.json) — Completed-run budget, best score and elapsed-time marker.
- [models/previous-runs/run3-pipeline-v3/showcase/history/stage3-seed0/training.jsonl](../models/previous-runs/run3-pipeline-v3/showcase/history/stage3-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### models/previous-runs/run3-pipeline-v3/showcase

- [models/previous-runs/run3-pipeline-v3/showcase/summary.json](../models/previous-runs/run3-pipeline-v3/showcase/summary.json) — Saved showcase selection, training lineage, held-out verdict and per-scenario comparison.
- [models/previous-runs/run3-pipeline-v3/showcase/test_baseline.json](../models/previous-runs/run3-pipeline-v3/showcase/test_baseline.json) — Saved showcase learned/baseline/comparison report.
- [models/previous-runs/run3-pipeline-v3/showcase/test_compare.json](../models/previous-runs/run3-pipeline-v3/showcase/test_compare.json) — Saved showcase learned/baseline/comparison report.
- [models/previous-runs/run3-pipeline-v3/showcase/test_learned.json](../models/previous-runs/run3-pipeline-v3/showcase/test_learned.json) — Saved showcase learned/baseline/comparison report.
- [models/previous-runs/run3-pipeline-v3/showcase/validation.json](../models/previous-runs/run3-pipeline-v3/showcase/validation.json) — Historical validation or copied best-selection evidence; inspect its schema/version.

### models/previous-runs/run3-pipeline-v3/stage1-seed0

- [models/previous-runs/run3-pipeline-v3/stage1-seed0/best.json](../models/previous-runs/run3-pipeline-v3/stage1-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/previous-runs/run3-pipeline-v3/stage1-seed0/best.pt](../models/previous-runs/run3-pipeline-v3/stage1-seed0/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.
- [models/previous-runs/run3-pipeline-v3/stage1-seed0/config.json](../models/previous-runs/run3-pipeline-v3/stage1-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/previous-runs/run3-pipeline-v3/stage1-seed0/evaluations.jsonl](../models/previous-runs/run3-pipeline-v3/stage1-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/previous-runs/run3-pipeline-v3/stage1-seed0/finished.json](../models/previous-runs/run3-pipeline-v3/stage1-seed0/finished.json) — Completed-run budget, best score and elapsed-time marker.
- [models/previous-runs/run3-pipeline-v3/stage1-seed0/training.jsonl](../models/previous-runs/run3-pipeline-v3/stage1-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### models/previous-runs/run3-pipeline-v3/stage2-seed0

- [models/previous-runs/run3-pipeline-v3/stage2-seed0/best.json](../models/previous-runs/run3-pipeline-v3/stage2-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/previous-runs/run3-pipeline-v3/stage2-seed0/best.pt](../models/previous-runs/run3-pipeline-v3/stage2-seed0/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.
- [models/previous-runs/run3-pipeline-v3/stage2-seed0/config.json](../models/previous-runs/run3-pipeline-v3/stage2-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/previous-runs/run3-pipeline-v3/stage2-seed0/evaluations.jsonl](../models/previous-runs/run3-pipeline-v3/stage2-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/previous-runs/run3-pipeline-v3/stage2-seed0/finished.json](../models/previous-runs/run3-pipeline-v3/stage2-seed0/finished.json) — Completed-run budget, best score and elapsed-time marker.
- [models/previous-runs/run3-pipeline-v3/stage2-seed0/training.jsonl](../models/previous-runs/run3-pipeline-v3/stage2-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### models/previous-runs/run3-pipeline-v3/stage3-seed0

- [models/previous-runs/run3-pipeline-v3/stage3-seed0/best.json](../models/previous-runs/run3-pipeline-v3/stage3-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/previous-runs/run3-pipeline-v3/stage3-seed0/best.pt](../models/previous-runs/run3-pipeline-v3/stage3-seed0/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.
- [models/previous-runs/run3-pipeline-v3/stage3-seed0/config.json](../models/previous-runs/run3-pipeline-v3/stage3-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/previous-runs/run3-pipeline-v3/stage3-seed0/evaluations.jsonl](../models/previous-runs/run3-pipeline-v3/stage3-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/previous-runs/run3-pipeline-v3/stage3-seed0/finished.json](../models/previous-runs/run3-pipeline-v3/stage3-seed0/finished.json) — Completed-run budget, best score and elapsed-time marker.
- [models/previous-runs/run3-pipeline-v3/stage3-seed0/training.jsonl](../models/previous-runs/run3-pipeline-v3/stage3-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### models/previous-runs/run3-pipeline-v3

- [models/previous-runs/run3-pipeline-v3/viewer-training.log](../models/previous-runs/run3-pipeline-v3/viewer-training.log) — Historical runtime/training text log.

### models/stage1-seed0

- [models/stage1-seed0/best.json](../models/stage1-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/stage1-seed0/best.pt](../models/stage1-seed0/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### models/stage1-seed0/checkpoints

- [models/stage1-seed0/checkpoints/step_000501760.pt](../models/stage1-seed0/checkpoints/step_000501760.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.
- [models/stage1-seed0/checkpoints/step_001000000.pt](../models/stage1-seed0/checkpoints/step_001000000.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### models/stage1-seed0

- [models/stage1-seed0/config.json](../models/stage1-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/stage1-seed0/evaluations.jsonl](../models/stage1-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/stage1-seed0/finished.json](../models/stage1-seed0/finished.json) — Completed-run budget, best score and elapsed-time marker.
- [models/stage1-seed0/latest.pt](../models/stage1-seed0/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.
- [models/stage1-seed0/training.jsonl](../models/stage1-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### models/stage2-seed0

- [models/stage2-seed0/best.json](../models/stage2-seed0/best.json) — Best-checkpoint selection metrics and step.
- [models/stage2-seed0/best.pt](../models/stage2-seed0/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.
- [models/stage2-seed0/config.json](../models/stage2-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [models/stage2-seed0/evaluations.jsonl](../models/stage2-seed0/evaluations.jsonl) — Selection evaluations, including per-scenario metrics.
- [models/stage2-seed0/latest.pt](../models/stage2-seed0/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.
- [models/stage2-seed0/training.jsonl](../models/stage2-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

## Runs

Development reports, early SwarmFlow checkpoints, local logs and legacy metric/event files.


### runs

- [runs/baseline-initial.json](../runs/baseline-initial.json) — Evaluation/report data; retains exact configuration and/or per-episode evidence.
- [runs/baseline.json](../runs/baseline.json) — Evaluation/report data; retains exact configuration and/or per-episode evidence.
- [runs/local-demo.log](../runs/local-demo.log) — Historical runtime/training text log.

### runs/local-mode-check

- [runs/local-mode-check/baseline.json](../runs/local-mode-check/baseline.json) — Evaluation/report data; retains exact configuration and/or per-episode evidence.
- [runs/local-mode-check/learned.json](../runs/local-mode-check/learned.json) — Evaluation/report data; retains exact configuration and/or per-episode evidence.

### runs/my-smoke

- [runs/my-smoke/config.json](../runs/my-smoke/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [runs/my-smoke/evaluation.json](../runs/my-smoke/evaluation.json) — Evaluation/report data; retains exact configuration and/or per-episode evidence.
- [runs/my-smoke/latest.pt](../runs/my-smoke/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.
- [runs/my-smoke/training.jsonl](../runs/my-smoke/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### runs

- [runs/no-avoidance.json](../runs/no-avoidance.json) — Evaluation/report data; retains exact configuration and/or per-episode evidence.
- [runs/scaling-16.json](../runs/scaling-16.json) — Evaluation/report data; retains exact configuration and/or per-episode evidence.

### runs/seed0

- [runs/seed0/events.out.tfevents.1789735407.Dheeraj.24956.0](../runs/seed0/events.out.tfevents.1789735407.Dheeraj.24956.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789735859.Dheeraj.26960.0](../runs/seed0/events.out.tfevents.1789735859.Dheeraj.26960.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789762827.Dheeraj.24288.0](../runs/seed0/events.out.tfevents.1789762827.Dheeraj.24288.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789766008.Dheeraj.12068.0](../runs/seed0/events.out.tfevents.1789766008.Dheeraj.12068.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789766145.Dheeraj.11716.0](../runs/seed0/events.out.tfevents.1789766145.Dheeraj.11716.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789766264.Dheeraj.26472.0](../runs/seed0/events.out.tfevents.1789766264.Dheeraj.26472.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789766499.Dheeraj.5744.0](../runs/seed0/events.out.tfevents.1789766499.Dheeraj.5744.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789766700.Dheeraj.2368.0](../runs/seed0/events.out.tfevents.1789766700.Dheeraj.2368.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789767127.Dheeraj.18152.0](../runs/seed0/events.out.tfevents.1789767127.Dheeraj.18152.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789769115.Dheeraj.18204.0](../runs/seed0/events.out.tfevents.1789769115.Dheeraj.18204.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789769547.Dheeraj.19168.0](../runs/seed0/events.out.tfevents.1789769547.Dheeraj.19168.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789772509.Dheeraj.6844.0](../runs/seed0/events.out.tfevents.1789772509.Dheeraj.6844.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789772883.Dheeraj.18316.0](../runs/seed0/events.out.tfevents.1789772883.Dheeraj.18316.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789773386.Dheeraj.24880.0](../runs/seed0/events.out.tfevents.1789773386.Dheeraj.24880.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789774897.Dheeraj.22640.0](../runs/seed0/events.out.tfevents.1789774897.Dheeraj.22640.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789775838.Dheeraj.6716.0](../runs/seed0/events.out.tfevents.1789775838.Dheeraj.6716.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789777216.Dheeraj.3128.0](../runs/seed0/events.out.tfevents.1789777216.Dheeraj.3128.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789778621.Dheeraj.21052.0](../runs/seed0/events.out.tfevents.1789778621.Dheeraj.21052.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789779467.Dheeraj.14000.0](../runs/seed0/events.out.tfevents.1789779467.Dheeraj.14000.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789781936.Dheeraj.25952.0](../runs/seed0/events.out.tfevents.1789781936.Dheeraj.25952.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789782328.Dheeraj.10748.0](../runs/seed0/events.out.tfevents.1789782328.Dheeraj.10748.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789957651.Dheeraj.26932.0](../runs/seed0/events.out.tfevents.1789957651.Dheeraj.26932.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789957820.Dheeraj.7072.0](../runs/seed0/events.out.tfevents.1789957820.Dheeraj.7072.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789957964.Dheeraj.10120.0](../runs/seed0/events.out.tfevents.1789957964.Dheeraj.10120.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789958423.Dheeraj.9900.0](../runs/seed0/events.out.tfevents.1789958423.Dheeraj.9900.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789958754.Dheeraj.19092.0](../runs/seed0/events.out.tfevents.1789958754.Dheeraj.19092.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789959056.Dheeraj.4348.0](../runs/seed0/events.out.tfevents.1789959056.Dheeraj.4348.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/events.out.tfevents.1789959501.Dheeraj.13416.0](../runs/seed0/events.out.tfevents.1789959501.Dheeraj.13416.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed0/metrics.csv](../runs/seed0/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### runs/seed1

- [runs/seed1/events.out.tfevents.1789736793.Dheeraj.17964.0](../runs/seed1/events.out.tfevents.1789736793.Dheeraj.17964.0) — Legacy TensorBoard event stream; not generated by current JSONL trainer.
- [runs/seed1/metrics.csv](../runs/seed1/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### runs/smoke-policy

- [runs/smoke-policy/config.json](../runs/smoke-policy/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [runs/smoke-policy/evaluation.json](../runs/smoke-policy/evaluation.json) — Evaluation/report data; retains exact configuration and/or per-episode evidence.
- [runs/smoke-policy/latest.pt](../runs/smoke-policy/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.
- [runs/smoke-policy/training.jsonl](../runs/smoke-policy/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

### runs/stage1-seed0

- [runs/stage1-seed0/config.json](../runs/stage1-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [runs/stage1-seed0/latest.pt](../runs/stage1-seed0/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.
- [runs/stage1-seed0/training.jsonl](../runs/stage1-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.
- [runs/stage1-seed0/validation.json](../runs/stage1-seed0/validation.json) — Historical validation or copied best-selection evidence; inspect its schema/version.

### runs/stage2-seed0

- [runs/stage2-seed0/config.json](../runs/stage2-seed0/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [runs/stage2-seed0/latest.pt](../runs/stage2-seed0/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.
- [runs/stage2-seed0/training.jsonl](../runs/stage2-seed0/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.
- [runs/stage2-seed0/validation.json](../runs/stage2-seed0/validation.json) — Historical validation or copied best-selection evidence; inspect its schema/version.

### runs/training-demo/stage1

- [runs/training-demo/stage1/config.json](../runs/training-demo/stage1/config.json) — Saved run environment, model and/or training settings; use for exact provenance.
- [runs/training-demo/stage1/latest.pt](../runs/training-demo/stage1/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.
- [runs/training-demo/stage1/training.jsonl](../runs/training-demo/stage1/training.jsonl) — Cloning rounds and/or PPO updates, one JSON object per line.

## Results

Earlier BTP1 archived/submitted packs, not current SwarmFlow benchmark outputs.


### results/archive/20260918_204915_baseline/data

- [results/archive/20260918_204915_baseline/data/metrics.csv](../results/archive/20260918_204915_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_204915_baseline/data/trajectory.csv](../results/archive/20260918_204915_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_204915_baseline/graphs

- [results/archive/20260918_204915_baseline/graphs/distance.png](../results/archive/20260918_204915_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_204915_baseline

- [results/archive/20260918_204915_baseline/result.json](../results/archive/20260918_204915_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_204915_baseline/result.md](../results/archive/20260918_204915_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_204929_baseline/data

- [results/archive/20260918_204929_baseline/data/metrics.csv](../results/archive/20260918_204929_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_204929_baseline/data/trajectory.csv](../results/archive/20260918_204929_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_204929_baseline/graphs

- [results/archive/20260918_204929_baseline/graphs/distance.png](../results/archive/20260918_204929_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_204929_baseline

- [results/archive/20260918_204929_baseline/result.json](../results/archive/20260918_204929_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_204929_baseline/result.md](../results/archive/20260918_204929_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_204947_baseline/data

- [results/archive/20260918_204947_baseline/data/metrics.csv](../results/archive/20260918_204947_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_204947_baseline/data/trajectory.csv](../results/archive/20260918_204947_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_204947_baseline/graphs

- [results/archive/20260918_204947_baseline/graphs/distance.png](../results/archive/20260918_204947_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_204947_baseline

- [results/archive/20260918_204947_baseline/result.json](../results/archive/20260918_204947_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_204947_baseline/result.md](../results/archive/20260918_204947_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_204955_baseline/data

- [results/archive/20260918_204955_baseline/data/metrics.csv](../results/archive/20260918_204955_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_204955_baseline/data/trajectory.csv](../results/archive/20260918_204955_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_204955_baseline/graphs

- [results/archive/20260918_204955_baseline/graphs/distance.png](../results/archive/20260918_204955_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_204955_baseline/images

- [results/archive/20260918_204955_baseline/images/test1.png](../results/archive/20260918_204955_baseline/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_204955_baseline

- [results/archive/20260918_204955_baseline/result.json](../results/archive/20260918_204955_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_204955_baseline/result.md](../results/archive/20260918_204955_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_205058_train/data

- [results/archive/20260918_205058_train/data/metrics.csv](../results/archive/20260918_205058_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_205058_train/data/trajectory.csv](../results/archive/20260918_205058_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_205058_train/graphs

- [results/archive/20260918_205058_train/graphs/distance.png](../results/archive/20260918_205058_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_205058_train/images

- [results/archive/20260918_205058_train/images/test1.png](../results/archive/20260918_205058_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_205058_train/models

- [results/archive/20260918_205058_train/models/latest.pt](../results/archive/20260918_205058_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_205058_train

- [results/archive/20260918_205058_train/result.json](../results/archive/20260918_205058_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_205058_train/result.md](../results/archive/20260918_205058_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_205225_baseline/data

- [results/archive/20260918_205225_baseline/data/metrics.csv](../results/archive/20260918_205225_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_205225_baseline/data/trajectory.csv](../results/archive/20260918_205225_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_205225_baseline/graphs

- [results/archive/20260918_205225_baseline/graphs/distance.png](../results/archive/20260918_205225_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_205225_baseline

- [results/archive/20260918_205225_baseline/result.json](../results/archive/20260918_205225_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_205225_baseline/result.md](../results/archive/20260918_205225_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_205335_train/data

- [results/archive/20260918_205335_train/data/metrics.csv](../results/archive/20260918_205335_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_205335_train/data/trajectory.csv](../results/archive/20260918_205335_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_205335_train/graphs

- [results/archive/20260918_205335_train/graphs/distance.png](../results/archive/20260918_205335_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_205335_train/images

- [results/archive/20260918_205335_train/images/test1.png](../results/archive/20260918_205335_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_205335_train/models

- [results/archive/20260918_205335_train/models/latest.pt](../results/archive/20260918_205335_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_205335_train

- [results/archive/20260918_205335_train/result.json](../results/archive/20260918_205335_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_205335_train/result.md](../results/archive/20260918_205335_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_205754_baseline/data

- [results/archive/20260918_205754_baseline/data/metrics.csv](../results/archive/20260918_205754_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_205754_baseline/data/trajectory.csv](../results/archive/20260918_205754_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_205754_baseline/graphs

- [results/archive/20260918_205754_baseline/graphs/distance.png](../results/archive/20260918_205754_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_205754_baseline

- [results/archive/20260918_205754_baseline/result.json](../results/archive/20260918_205754_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_205754_baseline/result.md](../results/archive/20260918_205754_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_210539_baseline/data

- [results/archive/20260918_210539_baseline/data/metrics.csv](../results/archive/20260918_210539_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_210539_baseline/data/trajectory.csv](../results/archive/20260918_210539_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_210539_baseline/graphs

- [results/archive/20260918_210539_baseline/graphs/distance.png](../results/archive/20260918_210539_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_210539_baseline/images

- [results/archive/20260918_210539_baseline/images/test1.png](../results/archive/20260918_210539_baseline/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_210539_baseline

- [results/archive/20260918_210539_baseline/result.json](../results/archive/20260918_210539_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_210539_baseline/result.md](../results/archive/20260918_210539_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_210735_baseline/data

- [results/archive/20260918_210735_baseline/data/metrics.csv](../results/archive/20260918_210735_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_210735_baseline/data/trajectory.csv](../results/archive/20260918_210735_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_210735_baseline/graphs

- [results/archive/20260918_210735_baseline/graphs/distance.png](../results/archive/20260918_210735_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_210735_baseline/images

- [results/archive/20260918_210735_baseline/images/test2.png](../results/archive/20260918_210735_baseline/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_210735_baseline

- [results/archive/20260918_210735_baseline/result.json](../results/archive/20260918_210735_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_210735_baseline/result.md](../results/archive/20260918_210735_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_210850_train/data

- [results/archive/20260918_210850_train/data/metrics.csv](../results/archive/20260918_210850_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_210850_train/data/trajectory.csv](../results/archive/20260918_210850_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_210850_train/graphs

- [results/archive/20260918_210850_train/graphs/distance.png](../results/archive/20260918_210850_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_210850_train/images

- [results/archive/20260918_210850_train/images/test1.png](../results/archive/20260918_210850_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_210850_train/models

- [results/archive/20260918_210850_train/models/latest.pt](../results/archive/20260918_210850_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_210850_train

- [results/archive/20260918_210850_train/result.json](../results/archive/20260918_210850_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_210850_train/result.md](../results/archive/20260918_210850_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_211217_baseline/data

- [results/archive/20260918_211217_baseline/data/metrics.csv](../results/archive/20260918_211217_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_211217_baseline/data/trajectory.csv](../results/archive/20260918_211217_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_211217_baseline/graphs

- [results/archive/20260918_211217_baseline/graphs/distance.png](../results/archive/20260918_211217_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_211217_baseline

- [results/archive/20260918_211217_baseline/result.json](../results/archive/20260918_211217_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_211217_baseline/result.md](../results/archive/20260918_211217_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_211250_baseline/data

- [results/archive/20260918_211250_baseline/data/metrics.csv](../results/archive/20260918_211250_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_211250_baseline/data/trajectory.csv](../results/archive/20260918_211250_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_211250_baseline/graphs

- [results/archive/20260918_211250_baseline/graphs/distance.png](../results/archive/20260918_211250_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_211250_baseline

- [results/archive/20260918_211250_baseline/result.json](../results/archive/20260918_211250_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_211250_baseline/result.md](../results/archive/20260918_211250_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_211413_train/data

- [results/archive/20260918_211413_train/data/metrics.csv](../results/archive/20260918_211413_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_211413_train/data/trajectory.csv](../results/archive/20260918_211413_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_211413_train/graphs

- [results/archive/20260918_211413_train/graphs/distance.png](../results/archive/20260918_211413_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_211413_train/images

- [results/archive/20260918_211413_train/images/test1.png](../results/archive/20260918_211413_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_211413_train/images/test2.png](../results/archive/20260918_211413_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_211413_train/images/test3.png](../results/archive/20260918_211413_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_211413_train/models

- [results/archive/20260918_211413_train/models/latest.pt](../results/archive/20260918_211413_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_211413_train

- [results/archive/20260918_211413_train/result.json](../results/archive/20260918_211413_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_211413_train/result.md](../results/archive/20260918_211413_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_211515_baseline/data

- [results/archive/20260918_211515_baseline/data/metrics.csv](../results/archive/20260918_211515_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_211515_baseline/data/trajectory.csv](../results/archive/20260918_211515_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_211515_baseline/graphs

- [results/archive/20260918_211515_baseline/graphs/distance.png](../results/archive/20260918_211515_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_211515_baseline

- [results/archive/20260918_211515_baseline/result.json](../results/archive/20260918_211515_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_211515_baseline/result.md](../results/archive/20260918_211515_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_211605_train/data

- [results/archive/20260918_211605_train/data/metrics.csv](../results/archive/20260918_211605_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_211605_train/data/trajectory.csv](../results/archive/20260918_211605_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_211605_train/graphs

- [results/archive/20260918_211605_train/graphs/distance.png](../results/archive/20260918_211605_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_211605_train/images

- [results/archive/20260918_211605_train/images/test1.png](../results/archive/20260918_211605_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_211605_train/images/test2.png](../results/archive/20260918_211605_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_211605_train/images/test3.png](../results/archive/20260918_211605_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_211605_train/models

- [results/archive/20260918_211605_train/models/latest.pt](../results/archive/20260918_211605_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_211605_train

- [results/archive/20260918_211605_train/result.json](../results/archive/20260918_211605_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_211605_train/result.md](../results/archive/20260918_211605_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_211650_baseline/data

- [results/archive/20260918_211650_baseline/data/metrics.csv](../results/archive/20260918_211650_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_211650_baseline/data/trajectory.csv](../results/archive/20260918_211650_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_211650_baseline/graphs

- [results/archive/20260918_211650_baseline/graphs/distance.png](../results/archive/20260918_211650_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_211650_baseline

- [results/archive/20260918_211650_baseline/result.json](../results/archive/20260918_211650_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_211650_baseline/result.md](../results/archive/20260918_211650_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_211805_train/data

- [results/archive/20260918_211805_train/data/metrics.csv](../results/archive/20260918_211805_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_211805_train/data/trajectory.csv](../results/archive/20260918_211805_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_211805_train/graphs

- [results/archive/20260918_211805_train/graphs/distance.png](../results/archive/20260918_211805_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_211805_train/images

- [results/archive/20260918_211805_train/images/test1.png](../results/archive/20260918_211805_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_211805_train/images/test2.png](../results/archive/20260918_211805_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_211805_train/images/test3.png](../results/archive/20260918_211805_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_211805_train/models

- [results/archive/20260918_211805_train/models/latest.pt](../results/archive/20260918_211805_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_211805_train

- [results/archive/20260918_211805_train/result.json](../results/archive/20260918_211805_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_211805_train/result.md](../results/archive/20260918_211805_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_211855_baseline/data

- [results/archive/20260918_211855_baseline/data/metrics.csv](../results/archive/20260918_211855_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_211855_baseline/data/trajectory.csv](../results/archive/20260918_211855_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_211855_baseline/graphs

- [results/archive/20260918_211855_baseline/graphs/distance.png](../results/archive/20260918_211855_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_211855_baseline

- [results/archive/20260918_211855_baseline/result.json](../results/archive/20260918_211855_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_211855_baseline/result.md](../results/archive/20260918_211855_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_211905_baseline/data

- [results/archive/20260918_211905_baseline/data/metrics.csv](../results/archive/20260918_211905_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_211905_baseline/data/trajectory.csv](../results/archive/20260918_211905_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_211905_baseline/graphs

- [results/archive/20260918_211905_baseline/graphs/distance.png](../results/archive/20260918_211905_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_211905_baseline

- [results/archive/20260918_211905_baseline/result.json](../results/archive/20260918_211905_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_211905_baseline/result.md](../results/archive/20260918_211905_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_212159_train/data

- [results/archive/20260918_212159_train/data/metrics.csv](../results/archive/20260918_212159_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_212159_train/data/trajectory.csv](../results/archive/20260918_212159_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_212159_train/graphs

- [results/archive/20260918_212159_train/graphs/distance.png](../results/archive/20260918_212159_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_212159_train/images

- [results/archive/20260918_212159_train/images/test1.png](../results/archive/20260918_212159_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_212159_train/images/test2.png](../results/archive/20260918_212159_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_212159_train/images/test3.png](../results/archive/20260918_212159_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_212159_train/models

- [results/archive/20260918_212159_train/models/latest.pt](../results/archive/20260918_212159_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_212159_train

- [results/archive/20260918_212159_train/result.json](../results/archive/20260918_212159_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_212159_train/result.md](../results/archive/20260918_212159_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_212328_baseline/data

- [results/archive/20260918_212328_baseline/data/metrics.csv](../results/archive/20260918_212328_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_212328_baseline/data/trajectory.csv](../results/archive/20260918_212328_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_212328_baseline/graphs

- [results/archive/20260918_212328_baseline/graphs/distance.png](../results/archive/20260918_212328_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_212328_baseline

- [results/archive/20260918_212328_baseline/result.json](../results/archive/20260918_212328_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_212328_baseline/result.md](../results/archive/20260918_212328_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_212351_baseline/data

- [results/archive/20260918_212351_baseline/data/metrics.csv](../results/archive/20260918_212351_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_212351_baseline/data/trajectory.csv](../results/archive/20260918_212351_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_212351_baseline/graphs

- [results/archive/20260918_212351_baseline/graphs/distance.png](../results/archive/20260918_212351_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_212351_baseline

- [results/archive/20260918_212351_baseline/result.json](../results/archive/20260918_212351_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_212351_baseline/result.md](../results/archive/20260918_212351_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_212518_train/data

- [results/archive/20260918_212518_train/data/metrics.csv](../results/archive/20260918_212518_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_212518_train/data/trajectory.csv](../results/archive/20260918_212518_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_212518_train/graphs

- [results/archive/20260918_212518_train/graphs/distance.png](../results/archive/20260918_212518_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_212518_train/images

- [results/archive/20260918_212518_train/images/test1.png](../results/archive/20260918_212518_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_212518_train/images/test2.png](../results/archive/20260918_212518_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_212518_train/images/test3.png](../results/archive/20260918_212518_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_212518_train/models

- [results/archive/20260918_212518_train/models/latest.pt](../results/archive/20260918_212518_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_212518_train

- [results/archive/20260918_212518_train/result.json](../results/archive/20260918_212518_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_212518_train/result.md](../results/archive/20260918_212518_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_212840_train/data

- [results/archive/20260918_212840_train/data/metrics.csv](../results/archive/20260918_212840_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_212840_train/data/trajectory.csv](../results/archive/20260918_212840_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_212840_train/graphs

- [results/archive/20260918_212840_train/graphs/distance.png](../results/archive/20260918_212840_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_212840_train/images

- [results/archive/20260918_212840_train/images/test1.png](../results/archive/20260918_212840_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_212840_train/models

- [results/archive/20260918_212840_train/models/latest.pt](../results/archive/20260918_212840_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_212840_train

- [results/archive/20260918_212840_train/result.json](../results/archive/20260918_212840_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_212840_train/result.md](../results/archive/20260918_212840_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_213151_prod/data

- [results/archive/20260918_213151_prod/data/metrics.csv](../results/archive/20260918_213151_prod/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_213151_prod/data/trajectory.csv](../results/archive/20260918_213151_prod/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_213151_prod/graphs

- [results/archive/20260918_213151_prod/graphs/distance.png](../results/archive/20260918_213151_prod/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_213151_prod/images

- [results/archive/20260918_213151_prod/images/test1.png](../results/archive/20260918_213151_prod/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_213151_prod/images/test2.png](../results/archive/20260918_213151_prod/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_213151_prod/images/test3.png](../results/archive/20260918_213151_prod/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_213151_prod/models

- [results/archive/20260918_213151_prod/models/best.pt](../results/archive/20260918_213151_prod/models/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_213151_prod

- [results/archive/20260918_213151_prod/result.json](../results/archive/20260918_213151_prod/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_213151_prod/result.md](../results/archive/20260918_213151_prod/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_213356_train/data

- [results/archive/20260918_213356_train/data/metrics.csv](../results/archive/20260918_213356_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_213356_train/data/trajectory.csv](../results/archive/20260918_213356_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_213356_train/graphs

- [results/archive/20260918_213356_train/graphs/distance.png](../results/archive/20260918_213356_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_213356_train/images

- [results/archive/20260918_213356_train/images/test1.png](../results/archive/20260918_213356_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_213356_train/images/test2.png](../results/archive/20260918_213356_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_213356_train/images/test3.png](../results/archive/20260918_213356_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_213356_train/models

- [results/archive/20260918_213356_train/models/latest.pt](../results/archive/20260918_213356_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_213356_train

- [results/archive/20260918_213356_train/result.json](../results/archive/20260918_213356_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_213356_train/result.md](../results/archive/20260918_213356_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_220047_baseline/data

- [results/archive/20260918_220047_baseline/data/metrics.csv](../results/archive/20260918_220047_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_220047_baseline/data/trajectory.csv](../results/archive/20260918_220047_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_220047_baseline/graphs

- [results/archive/20260918_220047_baseline/graphs/distance.png](../results/archive/20260918_220047_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_220047_baseline

- [results/archive/20260918_220047_baseline/result.json](../results/archive/20260918_220047_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_220047_baseline/result.md](../results/archive/20260918_220047_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_220548_train/data

- [results/archive/20260918_220548_train/data/metrics.csv](../results/archive/20260918_220548_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_220548_train/data/trajectory.csv](../results/archive/20260918_220548_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_220548_train/graphs

- [results/archive/20260918_220548_train/graphs/distance.png](../results/archive/20260918_220548_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_220548_train/images

- [results/archive/20260918_220548_train/images/test1.png](../results/archive/20260918_220548_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_220548_train/images/test2.png](../results/archive/20260918_220548_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_220548_train/images/test3.png](../results/archive/20260918_220548_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_220548_train/models

- [results/archive/20260918_220548_train/models/latest.pt](../results/archive/20260918_220548_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_220548_train

- [results/archive/20260918_220548_train/result.json](../results/archive/20260918_220548_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_220548_train/result.md](../results/archive/20260918_220548_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_220650_baseline/data

- [results/archive/20260918_220650_baseline/data/metrics.csv](../results/archive/20260918_220650_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_220650_baseline/data/trajectory.csv](../results/archive/20260918_220650_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_220650_baseline/graphs

- [results/archive/20260918_220650_baseline/graphs/distance.png](../results/archive/20260918_220650_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_220650_baseline/images

- [results/archive/20260918_220650_baseline/images/test1.png](../results/archive/20260918_220650_baseline/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_220650_baseline/images/test2.png](../results/archive/20260918_220650_baseline/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_220650_baseline/images/test3.png](../results/archive/20260918_220650_baseline/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_220650_baseline

- [results/archive/20260918_220650_baseline/result.json](../results/archive/20260918_220650_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_220650_baseline/result.md](../results/archive/20260918_220650_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_220806_baseline/data

- [results/archive/20260918_220806_baseline/data/metrics.csv](../results/archive/20260918_220806_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_220806_baseline/data/trajectory.csv](../results/archive/20260918_220806_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_220806_baseline/graphs

- [results/archive/20260918_220806_baseline/graphs/distance.png](../results/archive/20260918_220806_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_220806_baseline/images

- [results/archive/20260918_220806_baseline/images/test1.png](../results/archive/20260918_220806_baseline/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_220806_baseline/images/test2.png](../results/archive/20260918_220806_baseline/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_220806_baseline/images/test3.png](../results/archive/20260918_220806_baseline/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_220806_baseline

- [results/archive/20260918_220806_baseline/result.json](../results/archive/20260918_220806_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_220806_baseline/result.md](../results/archive/20260918_220806_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_220900_baseline/data

- [results/archive/20260918_220900_baseline/data/metrics.csv](../results/archive/20260918_220900_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_220900_baseline/data/trajectory.csv](../results/archive/20260918_220900_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_220900_baseline/graphs

- [results/archive/20260918_220900_baseline/graphs/distance.png](../results/archive/20260918_220900_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_220900_baseline/images

- [results/archive/20260918_220900_baseline/images/test1.png](../results/archive/20260918_220900_baseline/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_220900_baseline/images/test2.png](../results/archive/20260918_220900_baseline/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_220900_baseline/images/test3.png](../results/archive/20260918_220900_baseline/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_220900_baseline

- [results/archive/20260918_220900_baseline/result.json](../results/archive/20260918_220900_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_220900_baseline/result.md](../results/archive/20260918_220900_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_220941_baseline/data

- [results/archive/20260918_220941_baseline/data/metrics.csv](../results/archive/20260918_220941_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_220941_baseline/data/trajectory.csv](../results/archive/20260918_220941_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_220941_baseline/graphs

- [results/archive/20260918_220941_baseline/graphs/distance.png](../results/archive/20260918_220941_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_220941_baseline/images

- [results/archive/20260918_220941_baseline/images/test1.png](../results/archive/20260918_220941_baseline/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_220941_baseline/images/test2.png](../results/archive/20260918_220941_baseline/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_220941_baseline/images/test3.png](../results/archive/20260918_220941_baseline/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_220941_baseline

- [results/archive/20260918_220941_baseline/result.json](../results/archive/20260918_220941_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_220941_baseline/result.md](../results/archive/20260918_220941_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_221021_baseline/data

- [results/archive/20260918_221021_baseline/data/metrics.csv](../results/archive/20260918_221021_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_221021_baseline/data/trajectory.csv](../results/archive/20260918_221021_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_221021_baseline/graphs

- [results/archive/20260918_221021_baseline/graphs/distance.png](../results/archive/20260918_221021_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_221021_baseline/images

- [results/archive/20260918_221021_baseline/images/test1.png](../results/archive/20260918_221021_baseline/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_221021_baseline/images/test2.png](../results/archive/20260918_221021_baseline/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_221021_baseline/images/test3.png](../results/archive/20260918_221021_baseline/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_221021_baseline

- [results/archive/20260918_221021_baseline/result.json](../results/archive/20260918_221021_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_221021_baseline/result.md](../results/archive/20260918_221021_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_221307_train/data

- [results/archive/20260918_221307_train/data/metrics.csv](../results/archive/20260918_221307_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_221307_train/data/trajectory.csv](../results/archive/20260918_221307_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_221307_train/graphs

- [results/archive/20260918_221307_train/graphs/distance.png](../results/archive/20260918_221307_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_221307_train/images

- [results/archive/20260918_221307_train/images/test1.png](../results/archive/20260918_221307_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_221307_train/images/test2.png](../results/archive/20260918_221307_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_221307_train/images/test3.png](../results/archive/20260918_221307_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_221307_train/models

- [results/archive/20260918_221307_train/models/latest.pt](../results/archive/20260918_221307_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_221307_train

- [results/archive/20260918_221307_train/result.json](../results/archive/20260918_221307_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_221307_train/result.md](../results/archive/20260918_221307_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_221343_prod/data

- [results/archive/20260918_221343_prod/data/metrics.csv](../results/archive/20260918_221343_prod/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_221343_prod/data/trajectory.csv](../results/archive/20260918_221343_prod/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_221343_prod/graphs

- [results/archive/20260918_221343_prod/graphs/distance.png](../results/archive/20260918_221343_prod/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_221343_prod/images

- [results/archive/20260918_221343_prod/images/test1.png](../results/archive/20260918_221343_prod/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_221343_prod/images/test2.png](../results/archive/20260918_221343_prod/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_221343_prod/images/test3.png](../results/archive/20260918_221343_prod/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_221343_prod/models

- [results/archive/20260918_221343_prod/models/best.pt](../results/archive/20260918_221343_prod/models/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_221343_prod

- [results/archive/20260918_221343_prod/result.json](../results/archive/20260918_221343_prod/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_221343_prod/result.md](../results/archive/20260918_221343_prod/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_221511_prod/data

- [results/archive/20260918_221511_prod/data/metrics.csv](../results/archive/20260918_221511_prod/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_221511_prod/data/trajectory.csv](../results/archive/20260918_221511_prod/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_221511_prod/graphs

- [results/archive/20260918_221511_prod/graphs/distance.png](../results/archive/20260918_221511_prod/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_221511_prod/images

- [results/archive/20260918_221511_prod/images/test1.png](../results/archive/20260918_221511_prod/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_221511_prod/models

- [results/archive/20260918_221511_prod/models/best.pt](../results/archive/20260918_221511_prod/models/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_221511_prod

- [results/archive/20260918_221511_prod/result.json](../results/archive/20260918_221511_prod/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_221511_prod/result.md](../results/archive/20260918_221511_prod/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_221811_prod/data

- [results/archive/20260918_221811_prod/data/metrics.csv](../results/archive/20260918_221811_prod/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_221811_prod/data/trajectory.csv](../results/archive/20260918_221811_prod/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_221811_prod/graphs

- [results/archive/20260918_221811_prod/graphs/distance.png](../results/archive/20260918_221811_prod/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_221811_prod/images

- [results/archive/20260918_221811_prod/images/test2.png](../results/archive/20260918_221811_prod/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_221811_prod/models

- [results/archive/20260918_221811_prod/models/best.pt](../results/archive/20260918_221811_prod/models/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_221811_prod

- [results/archive/20260918_221811_prod/result.json](../results/archive/20260918_221811_prod/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_221811_prod/result.md](../results/archive/20260918_221811_prod/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_222731_baseline/data

- [results/archive/20260918_222731_baseline/data/metrics.csv](../results/archive/20260918_222731_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_222731_baseline/data/trajectory.csv](../results/archive/20260918_222731_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_222731_baseline/graphs

- [results/archive/20260918_222731_baseline/graphs/distance.png](../results/archive/20260918_222731_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_222731_baseline

- [results/archive/20260918_222731_baseline/result.json](../results/archive/20260918_222731_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_222731_baseline/result.md](../results/archive/20260918_222731_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_223735_baseline/data

- [results/archive/20260918_223735_baseline/data/metrics.csv](../results/archive/20260918_223735_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_223735_baseline/data/trajectory.csv](../results/archive/20260918_223735_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_223735_baseline/graphs

- [results/archive/20260918_223735_baseline/graphs/distance.png](../results/archive/20260918_223735_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_223735_baseline/images

- [results/archive/20260918_223735_baseline/images/test2.png](../results/archive/20260918_223735_baseline/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_223735_baseline

- [results/archive/20260918_223735_baseline/result.json](../results/archive/20260918_223735_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_223735_baseline/result.md](../results/archive/20260918_223735_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_224027_baseline/data

- [results/archive/20260918_224027_baseline/data/metrics.csv](../results/archive/20260918_224027_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_224027_baseline/data/trajectory.csv](../results/archive/20260918_224027_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_224027_baseline/graphs

- [results/archive/20260918_224027_baseline/graphs/distance.png](../results/archive/20260918_224027_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_224027_baseline

- [results/archive/20260918_224027_baseline/result.json](../results/archive/20260918_224027_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_224027_baseline/result.md](../results/archive/20260918_224027_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_224618_baseline/data

- [results/archive/20260918_224618_baseline/data/metrics.csv](../results/archive/20260918_224618_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_224618_baseline/data/trajectory.csv](../results/archive/20260918_224618_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_224618_baseline/graphs

- [results/archive/20260918_224618_baseline/graphs/distance.png](../results/archive/20260918_224618_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_224618_baseline

- [results/archive/20260918_224618_baseline/result.json](../results/archive/20260918_224618_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_224618_baseline/result.md](../results/archive/20260918_224618_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_224628_baseline/data

- [results/archive/20260918_224628_baseline/data/metrics.csv](../results/archive/20260918_224628_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_224628_baseline/data/trajectory.csv](../results/archive/20260918_224628_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_224628_baseline/graphs

- [results/archive/20260918_224628_baseline/graphs/distance.png](../results/archive/20260918_224628_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_224628_baseline

- [results/archive/20260918_224628_baseline/result.json](../results/archive/20260918_224628_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_224628_baseline/result.md](../results/archive/20260918_224628_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_225350_baseline/data

- [results/archive/20260918_225350_baseline/data/metrics.csv](../results/archive/20260918_225350_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_225350_baseline/data/trajectory.csv](../results/archive/20260918_225350_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_225350_baseline/graphs

- [results/archive/20260918_225350_baseline/graphs/distance.png](../results/archive/20260918_225350_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_225350_baseline/images

- [results/archive/20260918_225350_baseline/images/test2.png](../results/archive/20260918_225350_baseline/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_225350_baseline

- [results/archive/20260918_225350_baseline/result.json](../results/archive/20260918_225350_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_225350_baseline/result.md](../results/archive/20260918_225350_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_225449_baseline/data

- [results/archive/20260918_225449_baseline/data/metrics.csv](../results/archive/20260918_225449_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_225449_baseline/data/trajectory.csv](../results/archive/20260918_225449_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_225449_baseline/graphs

- [results/archive/20260918_225449_baseline/graphs/distance.png](../results/archive/20260918_225449_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_225449_baseline/images

- [results/archive/20260918_225449_baseline/images/test3.png](../results/archive/20260918_225449_baseline/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_225449_baseline

- [results/archive/20260918_225449_baseline/result.json](../results/archive/20260918_225449_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_225449_baseline/result.md](../results/archive/20260918_225449_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_225635_baseline/data

- [results/archive/20260918_225635_baseline/data/metrics.csv](../results/archive/20260918_225635_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_225635_baseline/data/trajectory.csv](../results/archive/20260918_225635_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_225635_baseline/graphs

- [results/archive/20260918_225635_baseline/graphs/distance.png](../results/archive/20260918_225635_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_225635_baseline/images

- [results/archive/20260918_225635_baseline/images/test3.png](../results/archive/20260918_225635_baseline/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_225635_baseline

- [results/archive/20260918_225635_baseline/result.json](../results/archive/20260918_225635_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_225635_baseline/result.md](../results/archive/20260918_225635_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_230216_train/data

- [results/archive/20260918_230216_train/data/metrics.csv](../results/archive/20260918_230216_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_230216_train/data/trajectory.csv](../results/archive/20260918_230216_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_230216_train/graphs

- [results/archive/20260918_230216_train/graphs/distance.png](../results/archive/20260918_230216_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_230216_train/images

- [results/archive/20260918_230216_train/images/test1.png](../results/archive/20260918_230216_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_230216_train/images/test2.png](../results/archive/20260918_230216_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_230216_train/images/test3.png](../results/archive/20260918_230216_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_230216_train/models

- [results/archive/20260918_230216_train/models/latest.pt](../results/archive/20260918_230216_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_230216_train

- [results/archive/20260918_230216_train/result.json](../results/archive/20260918_230216_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_230216_train/result.md](../results/archive/20260918_230216_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_230341_train/data

- [results/archive/20260918_230341_train/data/metrics.csv](../results/archive/20260918_230341_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_230341_train/data/trajectory.csv](../results/archive/20260918_230341_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_230341_train/graphs

- [results/archive/20260918_230341_train/graphs/distance.png](../results/archive/20260918_230341_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_230341_train/images

- [results/archive/20260918_230341_train/images/test2.png](../results/archive/20260918_230341_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_230341_train/models

- [results/archive/20260918_230341_train/models/latest.pt](../results/archive/20260918_230341_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_230341_train

- [results/archive/20260918_230341_train/result.json](../results/archive/20260918_230341_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_230341_train/result.md](../results/archive/20260918_230341_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_230604_train/data

- [results/archive/20260918_230604_train/data/metrics.csv](../results/archive/20260918_230604_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_230604_train/data/trajectory.csv](../results/archive/20260918_230604_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_230604_train/graphs

- [results/archive/20260918_230604_train/graphs/distance.png](../results/archive/20260918_230604_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_230604_train/images

- [results/archive/20260918_230604_train/images/test1.png](../results/archive/20260918_230604_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_230604_train/models

- [results/archive/20260918_230604_train/models/latest.pt](../results/archive/20260918_230604_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_230604_train

- [results/archive/20260918_230604_train/result.json](../results/archive/20260918_230604_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_230604_train/result.md](../results/archive/20260918_230604_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_231033_train/data

- [results/archive/20260918_231033_train/data/metrics.csv](../results/archive/20260918_231033_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_231033_train/data/trajectory.csv](../results/archive/20260918_231033_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_231033_train/graphs

- [results/archive/20260918_231033_train/graphs/distance.png](../results/archive/20260918_231033_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_231033_train/images

- [results/archive/20260918_231033_train/images/test1.png](../results/archive/20260918_231033_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_231033_train/images/test2.png](../results/archive/20260918_231033_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_231033_train/images/test3.png](../results/archive/20260918_231033_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_231033_train/models

- [results/archive/20260918_231033_train/models/latest.pt](../results/archive/20260918_231033_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_231033_train

- [results/archive/20260918_231033_train/result.json](../results/archive/20260918_231033_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_231033_train/result.md](../results/archive/20260918_231033_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_232803_train/data

- [results/archive/20260918_232803_train/data/metrics.csv](../results/archive/20260918_232803_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_232803_train/data/trajectory.csv](../results/archive/20260918_232803_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_232803_train/graphs

- [results/archive/20260918_232803_train/graphs/distance.png](../results/archive/20260918_232803_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_232803_train/images

- [results/archive/20260918_232803_train/images/test1.png](../results/archive/20260918_232803_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_232803_train/images/test2.png](../results/archive/20260918_232803_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_232803_train/images/test3.png](../results/archive/20260918_232803_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_232803_train/models

- [results/archive/20260918_232803_train/models/latest.pt](../results/archive/20260918_232803_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_232803_train

- [results/archive/20260918_232803_train/result.json](../results/archive/20260918_232803_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_232803_train/result.md](../results/archive/20260918_232803_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_233620_baseline/data

- [results/archive/20260918_233620_baseline/data/metrics.csv](../results/archive/20260918_233620_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_233620_baseline/data/trajectory.csv](../results/archive/20260918_233620_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_233620_baseline/graphs

- [results/archive/20260918_233620_baseline/graphs/distance.png](../results/archive/20260918_233620_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_233620_baseline

- [results/archive/20260918_233620_baseline/result.json](../results/archive/20260918_233620_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_233620_baseline/result.md](../results/archive/20260918_233620_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_233638_baseline/data

- [results/archive/20260918_233638_baseline/data/metrics.csv](../results/archive/20260918_233638_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_233638_baseline/data/trajectory.csv](../results/archive/20260918_233638_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_233638_baseline/graphs

- [results/archive/20260918_233638_baseline/graphs/distance.png](../results/archive/20260918_233638_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_233638_baseline

- [results/archive/20260918_233638_baseline/result.json](../results/archive/20260918_233638_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_233638_baseline/result.md](../results/archive/20260918_233638_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_235124_baseline/data

- [results/archive/20260918_235124_baseline/data/metrics.csv](../results/archive/20260918_235124_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_235124_baseline/data/trajectory.csv](../results/archive/20260918_235124_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_235124_baseline/graphs

- [results/archive/20260918_235124_baseline/graphs/distance.png](../results/archive/20260918_235124_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_235124_baseline

- [results/archive/20260918_235124_baseline/result.json](../results/archive/20260918_235124_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_235124_baseline/result.md](../results/archive/20260918_235124_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260918_235206_train/data

- [results/archive/20260918_235206_train/data/metrics.csv](../results/archive/20260918_235206_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260918_235206_train/data/trajectory.csv](../results/archive/20260918_235206_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260918_235206_train/graphs

- [results/archive/20260918_235206_train/graphs/distance.png](../results/archive/20260918_235206_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_235206_train/images

- [results/archive/20260918_235206_train/images/test1.png](../results/archive/20260918_235206_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_235206_train/images/test2.png](../results/archive/20260918_235206_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260918_235206_train/images/test3.png](../results/archive/20260918_235206_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260918_235206_train/models

- [results/archive/20260918_235206_train/models/latest.pt](../results/archive/20260918_235206_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260918_235206_train

- [results/archive/20260918_235206_train/result.json](../results/archive/20260918_235206_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260918_235206_train/result.md](../results/archive/20260918_235206_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_000357_train/data

- [results/archive/20260919_000357_train/data/metrics.csv](../results/archive/20260919_000357_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_000357_train/data/trajectory.csv](../results/archive/20260919_000357_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_000357_train/graphs

- [results/archive/20260919_000357_train/graphs/distance.png](../results/archive/20260919_000357_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_000357_train/images

- [results/archive/20260919_000357_train/images/test1.png](../results/archive/20260919_000357_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260919_000357_train/images/test2.png](../results/archive/20260919_000357_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260919_000357_train/images/test3.png](../results/archive/20260919_000357_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_000357_train/models

- [results/archive/20260919_000357_train/models/latest.pt](../results/archive/20260919_000357_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260919_000357_train

- [results/archive/20260919_000357_train/result.json](../results/archive/20260919_000357_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_000357_train/result.md](../results/archive/20260919_000357_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_000405_train/data

- [results/archive/20260919_000405_train/data/metrics.csv](../results/archive/20260919_000405_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_000405_train/data/trajectory.csv](../results/archive/20260919_000405_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_000405_train/graphs

- [results/archive/20260919_000405_train/graphs/distance.png](../results/archive/20260919_000405_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_000405_train/images

- [results/archive/20260919_000405_train/images/test2.png](../results/archive/20260919_000405_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_000405_train/models

- [results/archive/20260919_000405_train/models/latest.pt](../results/archive/20260919_000405_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260919_000405_train

- [results/archive/20260919_000405_train/result.json](../results/archive/20260919_000405_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_000405_train/result.md](../results/archive/20260919_000405_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_000539_train/data

- [results/archive/20260919_000539_train/data/metrics.csv](../results/archive/20260919_000539_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_000539_train/data/trajectory.csv](../results/archive/20260919_000539_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_000539_train/graphs

- [results/archive/20260919_000539_train/graphs/distance.png](../results/archive/20260919_000539_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_000539_train/images

- [results/archive/20260919_000539_train/images/test2.png](../results/archive/20260919_000539_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_000539_train/models

- [results/archive/20260919_000539_train/models/latest.pt](../results/archive/20260919_000539_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260919_000539_train

- [results/archive/20260919_000539_train/result.json](../results/archive/20260919_000539_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_000539_train/result.md](../results/archive/20260919_000539_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_000816_prod/data

- [results/archive/20260919_000816_prod/data/metrics.csv](../results/archive/20260919_000816_prod/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_000816_prod/data/trajectory.csv](../results/archive/20260919_000816_prod/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_000816_prod/graphs

- [results/archive/20260919_000816_prod/graphs/distance.png](../results/archive/20260919_000816_prod/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_000816_prod/images

- [results/archive/20260919_000816_prod/images/test2.png](../results/archive/20260919_000816_prod/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_000816_prod/models

- [results/archive/20260919_000816_prod/models/best.pt](../results/archive/20260919_000816_prod/models/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260919_000816_prod

- [results/archive/20260919_000816_prod/result.json](../results/archive/20260919_000816_prod/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_000816_prod/result.md](../results/archive/20260919_000816_prod/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_001249_baseline/data

- [results/archive/20260919_001249_baseline/data/metrics.csv](../results/archive/20260919_001249_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_001249_baseline/data/trajectory.csv](../results/archive/20260919_001249_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_001249_baseline/graphs

- [results/archive/20260919_001249_baseline/graphs/distance.png](../results/archive/20260919_001249_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_001249_baseline/images

- [results/archive/20260919_001249_baseline/images/test2.png](../results/archive/20260919_001249_baseline/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_001249_baseline

- [results/archive/20260919_001249_baseline/result.json](../results/archive/20260919_001249_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_001249_baseline/result.md](../results/archive/20260919_001249_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_001458_baseline/data

- [results/archive/20260919_001458_baseline/data/metrics.csv](../results/archive/20260919_001458_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_001458_baseline/data/trajectory.csv](../results/archive/20260919_001458_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_001458_baseline/graphs

- [results/archive/20260919_001458_baseline/graphs/distance.png](../results/archive/20260919_001458_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_001458_baseline

- [results/archive/20260919_001458_baseline/result.json](../results/archive/20260919_001458_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_001458_baseline/result.md](../results/archive/20260919_001458_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_002633_train/data

- [results/archive/20260919_002633_train/data/metrics.csv](../results/archive/20260919_002633_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_002633_train/data/trajectory.csv](../results/archive/20260919_002633_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_002633_train/graphs

- [results/archive/20260919_002633_train/graphs/distance.png](../results/archive/20260919_002633_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_002633_train/images

- [results/archive/20260919_002633_train/images/test1.png](../results/archive/20260919_002633_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260919_002633_train/images/test2.png](../results/archive/20260919_002633_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260919_002633_train/images/test3.png](../results/archive/20260919_002633_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_002633_train/models

- [results/archive/20260919_002633_train/models/latest.pt](../results/archive/20260919_002633_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260919_002633_train

- [results/archive/20260919_002633_train/result.json](../results/archive/20260919_002633_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_002633_train/result.md](../results/archive/20260919_002633_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_002646_train/data

- [results/archive/20260919_002646_train/data/metrics.csv](../results/archive/20260919_002646_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_002646_train/data/trajectory.csv](../results/archive/20260919_002646_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_002646_train/graphs

- [results/archive/20260919_002646_train/graphs/distance.png](../results/archive/20260919_002646_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_002646_train/images

- [results/archive/20260919_002646_train/images/test2.png](../results/archive/20260919_002646_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_002646_train/models

- [results/archive/20260919_002646_train/models/latest.pt](../results/archive/20260919_002646_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260919_002646_train

- [results/archive/20260919_002646_train/result.json](../results/archive/20260919_002646_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_002646_train/result.md](../results/archive/20260919_002646_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_003112_prod/data

- [results/archive/20260919_003112_prod/data/metrics.csv](../results/archive/20260919_003112_prod/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_003112_prod/data/trajectory.csv](../results/archive/20260919_003112_prod/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_003112_prod/graphs

- [results/archive/20260919_003112_prod/graphs/distance.png](../results/archive/20260919_003112_prod/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_003112_prod/images

- [results/archive/20260919_003112_prod/images/test2.png](../results/archive/20260919_003112_prod/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_003112_prod/models

- [results/archive/20260919_003112_prod/models/best.pt](../results/archive/20260919_003112_prod/models/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260919_003112_prod

- [results/archive/20260919_003112_prod/result.json](../results/archive/20260919_003112_prod/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_003112_prod/result.md](../results/archive/20260919_003112_prod/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_004509_train/data

- [results/archive/20260919_004509_train/data/metrics.csv](../results/archive/20260919_004509_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_004509_train/data/trajectory.csv](../results/archive/20260919_004509_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_004509_train/graphs

- [results/archive/20260919_004509_train/graphs/distance.png](../results/archive/20260919_004509_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_004509_train/images

- [results/archive/20260919_004509_train/images/test1.png](../results/archive/20260919_004509_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260919_004509_train/images/test2.png](../results/archive/20260919_004509_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260919_004509_train/images/test3.png](../results/archive/20260919_004509_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_004509_train/models

- [results/archive/20260919_004509_train/models/latest.pt](../results/archive/20260919_004509_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260919_004509_train

- [results/archive/20260919_004509_train/result.json](../results/archive/20260919_004509_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_004509_train/result.md](../results/archive/20260919_004509_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_004642_train/data

- [results/archive/20260919_004642_train/data/metrics.csv](../results/archive/20260919_004642_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_004642_train/data/trajectory.csv](../results/archive/20260919_004642_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_004642_train/graphs

- [results/archive/20260919_004642_train/graphs/distance.png](../results/archive/20260919_004642_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_004642_train/images

- [results/archive/20260919_004642_train/images/test2.png](../results/archive/20260919_004642_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_004642_train/models

- [results/archive/20260919_004642_train/models/latest.pt](../results/archive/20260919_004642_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260919_004642_train

- [results/archive/20260919_004642_train/result.json](../results/archive/20260919_004642_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_004642_train/result.md](../results/archive/20260919_004642_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_004820_prod/data

- [results/archive/20260919_004820_prod/data/metrics.csv](../results/archive/20260919_004820_prod/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_004820_prod/data/trajectory.csv](../results/archive/20260919_004820_prod/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_004820_prod/graphs

- [results/archive/20260919_004820_prod/graphs/distance.png](../results/archive/20260919_004820_prod/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_004820_prod/images

- [results/archive/20260919_004820_prod/images/test2.png](../results/archive/20260919_004820_prod/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_004820_prod/models

- [results/archive/20260919_004820_prod/models/best.pt](../results/archive/20260919_004820_prod/models/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260919_004820_prod

- [results/archive/20260919_004820_prod/result.json](../results/archive/20260919_004820_prod/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_004820_prod/result.md](../results/archive/20260919_004820_prod/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_010338_train/data

- [results/archive/20260919_010338_train/data/metrics.csv](../results/archive/20260919_010338_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_010338_train/data/trajectory.csv](../results/archive/20260919_010338_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_010338_train/graphs

- [results/archive/20260919_010338_train/graphs/distance.png](../results/archive/20260919_010338_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_010338_train/images

- [results/archive/20260919_010338_train/images/test1.png](../results/archive/20260919_010338_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260919_010338_train/images/test2.png](../results/archive/20260919_010338_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260919_010338_train/images/test3.png](../results/archive/20260919_010338_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_010338_train/models

- [results/archive/20260919_010338_train/models/latest.pt](../results/archive/20260919_010338_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260919_010338_train

- [results/archive/20260919_010338_train/result.json](../results/archive/20260919_010338_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_010338_train/result.md](../results/archive/20260919_010338_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_010345_train/data

- [results/archive/20260919_010345_train/data/metrics.csv](../results/archive/20260919_010345_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_010345_train/data/trajectory.csv](../results/archive/20260919_010345_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_010345_train/graphs

- [results/archive/20260919_010345_train/graphs/distance.png](../results/archive/20260919_010345_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_010345_train/images

- [results/archive/20260919_010345_train/images/test2.png](../results/archive/20260919_010345_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_010345_train/models

- [results/archive/20260919_010345_train/models/latest.pt](../results/archive/20260919_010345_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260919_010345_train

- [results/archive/20260919_010345_train/result.json](../results/archive/20260919_010345_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_010345_train/result.md](../results/archive/20260919_010345_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_010413_train/data

- [results/archive/20260919_010413_train/data/metrics.csv](../results/archive/20260919_010413_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_010413_train/data/trajectory.csv](../results/archive/20260919_010413_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_010413_train/graphs

- [results/archive/20260919_010413_train/graphs/distance.png](../results/archive/20260919_010413_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_010413_train/images

- [results/archive/20260919_010413_train/images/test2.png](../results/archive/20260919_010413_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_010413_train/models

- [results/archive/20260919_010413_train/models/latest.pt](../results/archive/20260919_010413_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260919_010413_train

- [results/archive/20260919_010413_train/result.json](../results/archive/20260919_010413_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_010413_train/result.md](../results/archive/20260919_010413_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_011540_prod/data

- [results/archive/20260919_011540_prod/data/metrics.csv](../results/archive/20260919_011540_prod/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_011540_prod/data/trajectory.csv](../results/archive/20260919_011540_prod/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_011540_prod/graphs

- [results/archive/20260919_011540_prod/graphs/distance.png](../results/archive/20260919_011540_prod/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_011540_prod/images

- [results/archive/20260919_011540_prod/images/test2.png](../results/archive/20260919_011540_prod/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_011540_prod/models

- [results/archive/20260919_011540_prod/models/best.pt](../results/archive/20260919_011540_prod/models/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260919_011540_prod

- [results/archive/20260919_011540_prod/result.json](../results/archive/20260919_011540_prod/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_011540_prod/result.md](../results/archive/20260919_011540_prod/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_011941_baseline/data

- [results/archive/20260919_011941_baseline/data/metrics.csv](../results/archive/20260919_011941_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_011941_baseline/data/trajectory.csv](../results/archive/20260919_011941_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_011941_baseline/graphs

- [results/archive/20260919_011941_baseline/graphs/distance.png](../results/archive/20260919_011941_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_011941_baseline

- [results/archive/20260919_011941_baseline/result.json](../results/archive/20260919_011941_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_011941_baseline/result.md](../results/archive/20260919_011941_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_012317_baseline/data

- [results/archive/20260919_012317_baseline/data/metrics.csv](../results/archive/20260919_012317_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_012317_baseline/data/trajectory.csv](../results/archive/20260919_012317_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_012317_baseline/graphs

- [results/archive/20260919_012317_baseline/graphs/distance.png](../results/archive/20260919_012317_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_012317_baseline

- [results/archive/20260919_012317_baseline/result.json](../results/archive/20260919_012317_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_012317_baseline/result.md](../results/archive/20260919_012317_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_012604_baseline/data

- [results/archive/20260919_012604_baseline/data/metrics.csv](../results/archive/20260919_012604_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_012604_baseline/data/trajectory.csv](../results/archive/20260919_012604_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_012604_baseline/graphs

- [results/archive/20260919_012604_baseline/graphs/distance.png](../results/archive/20260919_012604_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_012604_baseline/images

- [results/archive/20260919_012604_baseline/images/test2.png](../results/archive/20260919_012604_baseline/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_012604_baseline

- [results/archive/20260919_012604_baseline/result.json](../results/archive/20260919_012604_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_012604_baseline/result.md](../results/archive/20260919_012604_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_013434_baseline/data

- [results/archive/20260919_013434_baseline/data/metrics.csv](../results/archive/20260919_013434_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_013434_baseline/data/trajectory.csv](../results/archive/20260919_013434_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_013434_baseline/graphs

- [results/archive/20260919_013434_baseline/graphs/distance.png](../results/archive/20260919_013434_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_013434_baseline

- [results/archive/20260919_013434_baseline/result.json](../results/archive/20260919_013434_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_013434_baseline/result.md](../results/archive/20260919_013434_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_013453_baseline/data

- [results/archive/20260919_013453_baseline/data/metrics.csv](../results/archive/20260919_013453_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_013453_baseline/data/trajectory.csv](../results/archive/20260919_013453_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_013453_baseline/graphs

- [results/archive/20260919_013453_baseline/graphs/distance.png](../results/archive/20260919_013453_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_013453_baseline/images

- [results/archive/20260919_013453_baseline/images/test1.png](../results/archive/20260919_013453_baseline/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260919_013453_baseline/images/test2.png](../results/archive/20260919_013453_baseline/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260919_013453_baseline/images/test3.png](../results/archive/20260919_013453_baseline/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_013453_baseline

- [results/archive/20260919_013453_baseline/result.json](../results/archive/20260919_013453_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_013453_baseline/result.md](../results/archive/20260919_013453_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_013921_train/data

- [results/archive/20260919_013921_train/data/metrics.csv](../results/archive/20260919_013921_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_013921_train/data/trajectory.csv](../results/archive/20260919_013921_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_013921_train/graphs

- [results/archive/20260919_013921_train/graphs/distance.png](../results/archive/20260919_013921_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_013921_train/images

- [results/archive/20260919_013921_train/images/test1.png](../results/archive/20260919_013921_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260919_013921_train/images/test2.png](../results/archive/20260919_013921_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260919_013921_train/images/test3.png](../results/archive/20260919_013921_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_013921_train/models

- [results/archive/20260919_013921_train/models/latest.pt](../results/archive/20260919_013921_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260919_013921_train

- [results/archive/20260919_013921_train/result.json](../results/archive/20260919_013921_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_013921_train/result.md](../results/archive/20260919_013921_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_014059_baseline/data

- [results/archive/20260919_014059_baseline/data/metrics.csv](../results/archive/20260919_014059_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_014059_baseline/data/trajectory.csv](../results/archive/20260919_014059_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_014059_baseline/graphs

- [results/archive/20260919_014059_baseline/graphs/distance.png](../results/archive/20260919_014059_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_014059_baseline/images

- [results/archive/20260919_014059_baseline/images/test2.png](../results/archive/20260919_014059_baseline/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_014059_baseline

- [results/archive/20260919_014059_baseline/result.json](../results/archive/20260919_014059_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_014059_baseline/result.md](../results/archive/20260919_014059_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_014115_baseline/data

- [results/archive/20260919_014115_baseline/data/metrics.csv](../results/archive/20260919_014115_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_014115_baseline/data/trajectory.csv](../results/archive/20260919_014115_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_014115_baseline/graphs

- [results/archive/20260919_014115_baseline/graphs/distance.png](../results/archive/20260919_014115_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_014115_baseline/images

- [results/archive/20260919_014115_baseline/images/test1.png](../results/archive/20260919_014115_baseline/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260919_014115_baseline/images/test2.png](../results/archive/20260919_014115_baseline/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260919_014115_baseline/images/test3.png](../results/archive/20260919_014115_baseline/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_014115_baseline

- [results/archive/20260919_014115_baseline/result.json](../results/archive/20260919_014115_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_014115_baseline/result.md](../results/archive/20260919_014115_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_014659_train/data

- [results/archive/20260919_014659_train/data/metrics.csv](../results/archive/20260919_014659_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_014659_train/data/trajectory.csv](../results/archive/20260919_014659_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_014659_train/graphs

- [results/archive/20260919_014659_train/graphs/distance.png](../results/archive/20260919_014659_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_014659_train/images

- [results/archive/20260919_014659_train/images/test1.png](../results/archive/20260919_014659_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260919_014659_train/images/test2.png](../results/archive/20260919_014659_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260919_014659_train/images/test3.png](../results/archive/20260919_014659_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_014659_train/models

- [results/archive/20260919_014659_train/models/latest.pt](../results/archive/20260919_014659_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260919_014659_train

- [results/archive/20260919_014659_train/result.json](../results/archive/20260919_014659_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_014659_train/result.md](../results/archive/20260919_014659_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_014834_train/data

- [results/archive/20260919_014834_train/data/metrics.csv](../results/archive/20260919_014834_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_014834_train/data/trajectory.csv](../results/archive/20260919_014834_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_014834_train/graphs

- [results/archive/20260919_014834_train/graphs/distance.png](../results/archive/20260919_014834_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_014834_train/images

- [results/archive/20260919_014834_train/images/test2.png](../results/archive/20260919_014834_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_014834_train/models

- [results/archive/20260919_014834_train/models/latest.pt](../results/archive/20260919_014834_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260919_014834_train

- [results/archive/20260919_014834_train/result.json](../results/archive/20260919_014834_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_014834_train/result.md](../results/archive/20260919_014834_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_015102_prod/data

- [results/archive/20260919_015102_prod/data/metrics.csv](../results/archive/20260919_015102_prod/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_015102_prod/data/trajectory.csv](../results/archive/20260919_015102_prod/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_015102_prod/graphs

- [results/archive/20260919_015102_prod/graphs/distance.png](../results/archive/20260919_015102_prod/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_015102_prod/images

- [results/archive/20260919_015102_prod/images/test2.png](../results/archive/20260919_015102_prod/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_015102_prod/models

- [results/archive/20260919_015102_prod/models/best.pt](../results/archive/20260919_015102_prod/models/best.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260919_015102_prod

- [results/archive/20260919_015102_prod/result.json](../results/archive/20260919_015102_prod/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_015102_prod/result.md](../results/archive/20260919_015102_prod/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_064626_baseline/data

- [results/archive/20260919_064626_baseline/data/metrics.csv](../results/archive/20260919_064626_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_064626_baseline/data/trajectory.csv](../results/archive/20260919_064626_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_064626_baseline/graphs

- [results/archive/20260919_064626_baseline/graphs/distance.png](../results/archive/20260919_064626_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_064626_baseline/images

- [results/archive/20260919_064626_baseline/images/test1.png](../results/archive/20260919_064626_baseline/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_064626_baseline

- [results/archive/20260919_064626_baseline/result.json](../results/archive/20260919_064626_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_064626_baseline/result.md](../results/archive/20260919_064626_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_064719_baseline/data

- [results/archive/20260919_064719_baseline/data/metrics.csv](../results/archive/20260919_064719_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_064719_baseline/data/trajectory.csv](../results/archive/20260919_064719_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_064719_baseline/graphs

- [results/archive/20260919_064719_baseline/graphs/distance.png](../results/archive/20260919_064719_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_064719_baseline/images

- [results/archive/20260919_064719_baseline/images/test2.png](../results/archive/20260919_064719_baseline/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_064719_baseline

- [results/archive/20260919_064719_baseline/result.json](../results/archive/20260919_064719_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_064719_baseline/result.md](../results/archive/20260919_064719_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_064742_baseline/data

- [results/archive/20260919_064742_baseline/data/metrics.csv](../results/archive/20260919_064742_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_064742_baseline/data/trajectory.csv](../results/archive/20260919_064742_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_064742_baseline/graphs

- [results/archive/20260919_064742_baseline/graphs/distance.png](../results/archive/20260919_064742_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_064742_baseline/images

- [results/archive/20260919_064742_baseline/images/test3.png](../results/archive/20260919_064742_baseline/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_064742_baseline

- [results/archive/20260919_064742_baseline/result.json](../results/archive/20260919_064742_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_064742_baseline/result.md](../results/archive/20260919_064742_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_065328_baseline/data

- [results/archive/20260919_065328_baseline/data/metrics.csv](../results/archive/20260919_065328_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_065328_baseline/data/trajectory.csv](../results/archive/20260919_065328_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_065328_baseline/graphs

- [results/archive/20260919_065328_baseline/graphs/distance.png](../results/archive/20260919_065328_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_065328_baseline

- [results/archive/20260919_065328_baseline/result.json](../results/archive/20260919_065328_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_065328_baseline/result.md](../results/archive/20260919_065328_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_065336_baseline/data

- [results/archive/20260919_065336_baseline/data/metrics.csv](../results/archive/20260919_065336_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_065336_baseline/data/trajectory.csv](../results/archive/20260919_065336_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_065336_baseline/graphs

- [results/archive/20260919_065336_baseline/graphs/distance.png](../results/archive/20260919_065336_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_065336_baseline/images

- [results/archive/20260919_065336_baseline/images/test1.png](../results/archive/20260919_065336_baseline/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260919_065336_baseline/images/test2.png](../results/archive/20260919_065336_baseline/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260919_065336_baseline/images/test3.png](../results/archive/20260919_065336_baseline/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_065336_baseline

- [results/archive/20260919_065336_baseline/result.json](../results/archive/20260919_065336_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_065336_baseline/result.md](../results/archive/20260919_065336_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_065420_baseline/data

- [results/archive/20260919_065420_baseline/data/metrics.csv](../results/archive/20260919_065420_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_065420_baseline/data/trajectory.csv](../results/archive/20260919_065420_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_065420_baseline/graphs

- [results/archive/20260919_065420_baseline/graphs/distance.png](../results/archive/20260919_065420_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_065420_baseline/images

- [results/archive/20260919_065420_baseline/images/test2.png](../results/archive/20260919_065420_baseline/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_065420_baseline

- [results/archive/20260919_065420_baseline/result.json](../results/archive/20260919_065420_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_065420_baseline/result.md](../results/archive/20260919_065420_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_065842_baseline/data

- [results/archive/20260919_065842_baseline/data/metrics.csv](../results/archive/20260919_065842_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_065842_baseline/data/trajectory.csv](../results/archive/20260919_065842_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_065842_baseline/graphs

- [results/archive/20260919_065842_baseline/graphs/distance.png](../results/archive/20260919_065842_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_065842_baseline/images

- [results/archive/20260919_065842_baseline/images/test2.png](../results/archive/20260919_065842_baseline/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_065842_baseline

- [results/archive/20260919_065842_baseline/result.json](../results/archive/20260919_065842_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_065842_baseline/result.md](../results/archive/20260919_065842_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_070109_baseline/data

- [results/archive/20260919_070109_baseline/data/metrics.csv](../results/archive/20260919_070109_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_070109_baseline/data/trajectory.csv](../results/archive/20260919_070109_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_070109_baseline/graphs

- [results/archive/20260919_070109_baseline/graphs/distance.png](../results/archive/20260919_070109_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_070109_baseline/images

- [results/archive/20260919_070109_baseline/images/test3.png](../results/archive/20260919_070109_baseline/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_070109_baseline

- [results/archive/20260919_070109_baseline/result.json](../results/archive/20260919_070109_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_070109_baseline/result.md](../results/archive/20260919_070109_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260919_070311_baseline/data

- [results/archive/20260919_070311_baseline/data/metrics.csv](../results/archive/20260919_070311_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260919_070311_baseline/data/trajectory.csv](../results/archive/20260919_070311_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260919_070311_baseline/graphs

- [results/archive/20260919_070311_baseline/graphs/distance.png](../results/archive/20260919_070311_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_070311_baseline/images

- [results/archive/20260919_070311_baseline/images/test1.png](../results/archive/20260919_070311_baseline/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260919_070311_baseline

- [results/archive/20260919_070311_baseline/result.json](../results/archive/20260919_070311_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260919_070311_baseline/result.md](../results/archive/20260919_070311_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260921_022123_baseline/data

- [results/archive/20260921_022123_baseline/data/metrics.csv](../results/archive/20260921_022123_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260921_022123_baseline/data/trajectory.csv](../results/archive/20260921_022123_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260921_022123_baseline/graphs

- [results/archive/20260921_022123_baseline/graphs/distance.png](../results/archive/20260921_022123_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260921_022123_baseline

- [results/archive/20260921_022123_baseline/result.json](../results/archive/20260921_022123_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260921_022123_baseline/result.md](../results/archive/20260921_022123_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260921_022818_train/data

- [results/archive/20260921_022818_train/data/metrics.csv](../results/archive/20260921_022818_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260921_022818_train/data/trajectory.csv](../results/archive/20260921_022818_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260921_022818_train/graphs

- [results/archive/20260921_022818_train/graphs/distance.png](../results/archive/20260921_022818_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260921_022818_train/images

- [results/archive/20260921_022818_train/images/test1.png](../results/archive/20260921_022818_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260921_022818_train/images/test2.png](../results/archive/20260921_022818_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260921_022818_train/images/test3.png](../results/archive/20260921_022818_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260921_022818_train/models

- [results/archive/20260921_022818_train/models/latest.pt](../results/archive/20260921_022818_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260921_022818_train

- [results/archive/20260921_022818_train/result.json](../results/archive/20260921_022818_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260921_022818_train/result.md](../results/archive/20260921_022818_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260921_023101_train/data

- [results/archive/20260921_023101_train/data/metrics.csv](../results/archive/20260921_023101_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260921_023101_train/data/trajectory.csv](../results/archive/20260921_023101_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260921_023101_train/graphs

- [results/archive/20260921_023101_train/graphs/distance.png](../results/archive/20260921_023101_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260921_023101_train/images

- [results/archive/20260921_023101_train/images/test1.png](../results/archive/20260921_023101_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260921_023101_train/images/test2.png](../results/archive/20260921_023101_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260921_023101_train/images/test3.png](../results/archive/20260921_023101_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260921_023101_train/models

- [results/archive/20260921_023101_train/models/latest.pt](../results/archive/20260921_023101_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260921_023101_train

- [results/archive/20260921_023101_train/result.json](../results/archive/20260921_023101_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260921_023101_train/result.md](../results/archive/20260921_023101_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260921_023327_train/data

- [results/archive/20260921_023327_train/data/metrics.csv](../results/archive/20260921_023327_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260921_023327_train/data/trajectory.csv](../results/archive/20260921_023327_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260921_023327_train/graphs

- [results/archive/20260921_023327_train/graphs/distance.png](../results/archive/20260921_023327_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260921_023327_train/images

- [results/archive/20260921_023327_train/images/test1.png](../results/archive/20260921_023327_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260921_023327_train/images/test2.png](../results/archive/20260921_023327_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260921_023327_train/images/test3.png](../results/archive/20260921_023327_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260921_023327_train/models

- [results/archive/20260921_023327_train/models/latest.pt](../results/archive/20260921_023327_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260921_023327_train

- [results/archive/20260921_023327_train/result.json](../results/archive/20260921_023327_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260921_023327_train/result.md](../results/archive/20260921_023327_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260921_024056_train/data

- [results/archive/20260921_024056_train/data/metrics.csv](../results/archive/20260921_024056_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260921_024056_train/data/trajectory.csv](../results/archive/20260921_024056_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260921_024056_train/graphs

- [results/archive/20260921_024056_train/graphs/distance.png](../results/archive/20260921_024056_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260921_024056_train/images

- [results/archive/20260921_024056_train/images/test1.png](../results/archive/20260921_024056_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260921_024056_train/images/test2.png](../results/archive/20260921_024056_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260921_024056_train/images/test3.png](../results/archive/20260921_024056_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260921_024056_train/models

- [results/archive/20260921_024056_train/models/latest.pt](../results/archive/20260921_024056_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260921_024056_train

- [results/archive/20260921_024056_train/result.json](../results/archive/20260921_024056_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260921_024056_train/result.md](../results/archive/20260921_024056_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260921_024558_baseline/data

- [results/archive/20260921_024558_baseline/data/metrics.csv](../results/archive/20260921_024558_baseline/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260921_024558_baseline/data/trajectory.csv](../results/archive/20260921_024558_baseline/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260921_024558_baseline/graphs

- [results/archive/20260921_024558_baseline/graphs/distance.png](../results/archive/20260921_024558_baseline/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260921_024558_baseline

- [results/archive/20260921_024558_baseline/result.json](../results/archive/20260921_024558_baseline/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260921_024558_baseline/result.md](../results/archive/20260921_024558_baseline/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260921_024637_train/data

- [results/archive/20260921_024637_train/data/metrics.csv](../results/archive/20260921_024637_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260921_024637_train/data/trajectory.csv](../results/archive/20260921_024637_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260921_024637_train/graphs

- [results/archive/20260921_024637_train/graphs/distance.png](../results/archive/20260921_024637_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260921_024637_train/images

- [results/archive/20260921_024637_train/images/test1.png](../results/archive/20260921_024637_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260921_024637_train/images/test2.png](../results/archive/20260921_024637_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260921_024637_train/images/test3.png](../results/archive/20260921_024637_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260921_024637_train/models

- [results/archive/20260921_024637_train/models/latest.pt](../results/archive/20260921_024637_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260921_024637_train

- [results/archive/20260921_024637_train/result.json](../results/archive/20260921_024637_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260921_024637_train/result.md](../results/archive/20260921_024637_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260921_025137_train/data

- [results/archive/20260921_025137_train/data/metrics.csv](../results/archive/20260921_025137_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260921_025137_train/data/trajectory.csv](../results/archive/20260921_025137_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260921_025137_train/graphs

- [results/archive/20260921_025137_train/graphs/distance.png](../results/archive/20260921_025137_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260921_025137_train/images

- [results/archive/20260921_025137_train/images/test1.png](../results/archive/20260921_025137_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260921_025137_train/images/test2.png](../results/archive/20260921_025137_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260921_025137_train/images/test3.png](../results/archive/20260921_025137_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260921_025137_train/models

- [results/archive/20260921_025137_train/models/latest.pt](../results/archive/20260921_025137_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260921_025137_train

- [results/archive/20260921_025137_train/result.json](../results/archive/20260921_025137_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260921_025137_train/result.md](../results/archive/20260921_025137_train/result.md) — Human-readable legacy BTP1 result pack.

### results/archive/20260921_025852_train/data

- [results/archive/20260921_025852_train/data/metrics.csv](../results/archive/20260921_025852_train/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/archive/20260921_025852_train/data/trajectory.csv](../results/archive/20260921_025852_train/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/archive/20260921_025852_train/graphs

- [results/archive/20260921_025852_train/graphs/distance.png](../results/archive/20260921_025852_train/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260921_025852_train/images

- [results/archive/20260921_025852_train/images/test1.png](../results/archive/20260921_025852_train/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260921_025852_train/images/test2.png](../results/archive/20260921_025852_train/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/archive/20260921_025852_train/images/test3.png](../results/archive/20260921_025852_train/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/archive/20260921_025852_train/models

- [results/archive/20260921_025852_train/models/latest.pt](../results/archive/20260921_025852_train/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/archive/20260921_025852_train

- [results/archive/20260921_025852_train/result.json](../results/archive/20260921_025852_train/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/archive/20260921_025852_train/result.md](../results/archive/20260921_025852_train/result.md) — Human-readable legacy BTP1 result pack.

### results/submit/data

- [results/submit/data/metrics.csv](../results/submit/data/metrics.csv) — Raw metrics or trajectory data; schema/provenance in history.md.
- [results/submit/data/trajectory.csv](../results/submit/data/trajectory.csv) — Raw metrics or trajectory data; schema/provenance in history.md.

### results/submit/graphs

- [results/submit/graphs/distance.png](../results/submit/graphs/distance.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/submit/images

- [results/submit/images/test1.png](../results/submit/images/test1.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/submit/images/test2.png](../results/submit/images/test2.png) — Saved image/plot artifact; quantitative claims require its companion report.
- [results/submit/images/test3.png](../results/submit/images/test3.png) — Saved image/plot artifact; quantitative claims require its companion report.

### results/submit/models

- [results/submit/models/latest.pt](../results/submit/models/latest.pt) — Binary checkpoint; metadata/provenance must match its companion run. Not read as executable pickle for this documentation.

### results/submit

- [results/submit/result.json](../results/submit/result.json) — Legacy BTP1 gate evaluation and original checkpoint provenance.
- [results/submit/result.md](../results/submit/result.md) — Human-readable legacy BTP1 result pack.

## Graphify metadata

Four hidden metadata files from the earlier graph tool. No runtime imports use them.

- [graphify-out/.graphify_labels.json](../graphify-out/.graphify_labels.json) — Evaluation/report data; retains exact configuration and/or per-episode evidence.
- [graphify-out/.graphify_labels.json.sig](../graphify-out/.graphify_labels.json.sig) — Legacy graph-tool metadata; labels do not supply missing source implementations.
- [graphify-out/.graphify_python](../graphify-out/.graphify_python) — Legacy graph-tool metadata; labels do not supply missing source implementations.
- [graphify-out/.graphify_root](../graphify-out/.graphify_root) — Legacy graph-tool metadata; labels do not supply missing source implementations.

## Generated directories and exclusions

- `.git/`: version-control metadata; no algorithmic context is attributed to its object database.
- `.venv/`: installed third-party interpreter/packages, declared by pyproject.toml and uv.lock.
- `.uv-cache/`: dependency download/build cache.
- `.pytest_cache/`: previous test-run bookkeeping, not a fresh pass result.
- `__pycache__/`, `*.egg-info/`, `build/` if generated: bytecode/packaging artifacts.
- `.DS_Store`: macOS folder display metadata.

## Coverage snapshot

Indexed **933 project files** outside the exclusions. Root/folder counts: docs: 6, graphify-out: 4, models: 138, results: 698, root: 10, runs: 57, src: 17, tests: 3.

If a cited historical path is absent from this inventory, see [missing context](history.md#missing-and-historical-only-context); it has not been silently reconstructed. Newly written logs/checkpoints can make this snapshot stale.
