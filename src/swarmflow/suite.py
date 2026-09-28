import json
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from .config import Config

TESTS_DIR = Path("runs/tests")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The eight scenarios behind the documented 24-episode baseline in docs/validation.md.
VALIDATION_SCENARIOS = ("hold", "follow", "crossing", "head_on", "double_crossing", "slalom", "turn", "gust")
OBSTACLE_SCENARIOS = ("crossing", "head_on", "double_crossing")
# Acceptance targets from docs/research-plan.md: no collisions, >=95% of events recovered within 5 s.
RECOVERY_LIMIT_S = 5.0
RECOVERY_RATE = 0.95
# Research-plan target "steady-state RMS slot error below 0.35 m", checked where no obstacle disturbs the formation.
FORMATION_SCENARIOS = ("hold", "follow", "turn")
FORMATION_RMSE_M = 0.35


@dataclass
class Options:
    episodes: int | None = None
    seed: int = 0
    retrain: bool = False
    keyword: str | None = None


@dataclass
class Result:
    verdict: str
    notes: list
    report: str | None = None


@dataclass(frozen=True, eq=False)
class Test:
    number: int
    slug: str
    title: str
    purpose: str
    criterion: str
    source: str
    estimate: str
    run: object
    scenarios: tuple = ()
    setup: dict | None = None
    watch: str | None = None


def _seeds(options, default, start):
    return range(start, start + (options.episodes or default))


def _evaluate(folder, name, scenarios, seeds, policy="baseline", checkpoint=None, config=None):
    from .evaluate import evaluate
    if checkpoint is not None:
        from .policy import LearnedController
        config = LearnedController(checkpoint).config
    output = folder / f"{name}.json"
    report = evaluate(config or Config(), list(scenarios), seeds, policy, checkpoint, str(output))
    return report, str(output)


def _acceptance(report, strict=True):
    """strict: research-plan targets (>=95% within 5 s); otherwise docs/validation.md (every event recovered)."""
    episodes = report["episodes"]
    failures = [e for e in episodes if e["failure"]]
    events = [(ev, e) for e in episodes for ev in e["events"]]
    recovered = [(ev, e) for ev, e in events if not e["failure"] and ev["recovery_s"] is not None]
    quick = [ev for ev, _ in recovered if ev["recovery_s"] <= RECOVERY_LIMIT_S]
    clearances = [e["min_surface_clearance_m"] for e in episodes if e["min_surface_clearance_m"] is not None]
    drifting = {name: row["mean_slot_rmse_m"] for name, row in report["summary"].items()
                if name in FORMATION_SCENARIOS and row["mean_slot_rmse_m"] >= FORMATION_RMSE_M}
    if strict:
        passed = not failures and (len(quick) >= RECOVERY_RATE * len(events))
        criterion = (f"criterion: no failures and >={RECOVERY_RATE:.0%} of events recovered within "
                     f"{RECOVERY_LIMIT_S:g} s (docs/research-plan.md)")
    else:
        passed = not failures and len(recovered) == len(events)
        criterion = "criterion: no failures and every event recovered (docs/validation.md)"
    passed = passed and not drifting
    criterion += f"; slot RMSE < {FORMATION_RMSE_M:g} m on {'/'.join(FORMATION_SCENARIOS)}"
    notes = [criterion, f"episodes without failure: {len(episodes) - len(failures)}/{len(episodes)}",
             f"events recovered: {len(recovered)}/{len(events)} "
             f"({len(quick)} within {RECOVERY_LIMIT_S:g} s)"]
    if clearances:
        notes.append(f"minimum surface clearance: {min(clearances):.4f} m")
    notes += [f"formation drift: {name} mean slot RMSE {value:.3f} m" for name, value in drifting.items()]
    notes += [f"failed: {e['scenario']} seed={e['seed']} ({e['failure']})" for e in failures[:5]]
    notes += [f"slow recovery: {e['scenario']} seed={e['seed']} {ev['name']} took {ev['recovery_s']:.2f} s"
              for ev, e in recovered if ev["recovery_s"] > RECOVERY_LIMIT_S][:5]
    return passed, notes


def _scenario_table(report):
    lines = [f"{'scenario':16s} {'ok':>7s} {'slot RMSE':>10s} {'recovered':>10s} {'clearance':>10s}"]
    for name, row in report["summary"].items():
        ok = f"{row['episodes'] - row['failures']}/{row['episodes']}"
        recovered = f"{row['recovered_events']}/{row['events']}" if row["events"] else "-"
        clearance = row["min_surface_clearance_m"]
        clearance = f"{clearance:.3f} m" if clearance is not None else "-"
        lines.append(f"{name:16s} {ok:>7s} {row['mean_slot_rmse_m']:>8.3f} m {recovered:>10s} {clearance:>10s}")
    return lines


def _baseline_test(scenarios, default_episodes, seed_start, config_changes=None, strict=True):
    def run(folder, options):
        config = replace(Config(), **(config_changes or {}))
        report, path = _evaluate(folder, "report", scenarios, _seeds(options, default_episodes, seed_start),
                                 config=config)
        passed, notes = _acceptance(report, strict)
        return Result("PASS" if passed else "FAIL", _scenario_table(report) + notes, path)
    return run


def run_unit_checks(folder, options):
    command = [sys.executable, "-m", "pytest", "-q"]
    if options.keyword:
        command += ["-k", options.keyword]
    if not (PROJECT_ROOT / "tests").is_dir():
        return Result("FAIL", [f"tests/ folder not found under {PROJECT_ROOT}"])
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return Result("PASS" if completed.returncode == 0 else "FAIL",
                  [f"pytest exit code {completed.returncode}" + (f" (-k {options.keyword})" if options.keyword else "")])


def run_no_avoidance(folder, options):
    report, path = _evaluate(folder, "report", OBSTACLE_SCENARIOS, _seeds(options, 3, 1000), policy="slot")
    episodes = report["episodes"]
    collided = sum(bool(e["collision"]) for e in episodes)
    notes = _scenario_table(report) + [
        f"slot-only controller collided in {collided}/{len(episodes)} obstacle episodes",
        "PASS means the obstacles are a real threat: ignoring them causes collisions."]
    return Result("PASS" if collided == len(episodes) else "FAIL", notes, path)


def _train_or_reuse(folder, options, **train_args):
    from .train import train
    checkpoint = folder / "latest.pt"
    if checkpoint.exists() and not options.retrain:
        print(f"Reusing existing checkpoint {checkpoint} (pass --retrain to train again)", flush=True)
        return checkpoint, True
    if checkpoint.exists():
        archived = folder.with_name(f"{folder.name}-old-{datetime.now():%Y%m%d-%H%M%S}")
        folder.rename(archived)
        print(f"Moved previous run to {archived}", flush=True)
    train(Config(), output=str(folder), seed=options.seed, **train_args)
    return checkpoint, False


def run_smoke_training(folder, options):
    checkpoint, reused = _train_or_reuse(folder, options, scenarios=["hold", "follow", "crossing"],
                                         total_steps=512, horizon=64, epochs=2, clone_steps=128, clone_epochs=3,
                                         eval_seeds=1)
    report, path = _evaluate(folder, "evaluation", ("follow", "crossing"), _seeds(options, 3, 1000),
                             policy="learned", checkpoint=str(checkpoint))
    notes = _scenario_table(report) + [
        f"checkpoint {'reused' if reused else 'trained'}, saved and reloaded: {checkpoint}",
        "PASS checks the training/checkpoint machinery only; a smoke policy is not expected to fly well."]
    return Result("PASS", notes, path)


def _learned_vs_baseline(folder, options, checkpoint, scenarios):
    seeds = _seeds(options, 20, 1000)
    learned, path = _evaluate(folder, "validation", scenarios, seeds, policy="learned",
                              checkpoint=str(checkpoint))
    _, baseline_path = _evaluate(folder, "baseline-same-seeds", scenarios, seeds,
                                 config=Config(**learned["config"]))
    from .evaluate import compare_reports
    comparison = compare_reports((baseline_path, path), str(folder / "compare.json"))
    passed, notes = _acceptance(learned)
    lines = ["learned policy:"] + _scenario_table(learned) + ["", "learned vs predictive baseline (same seeds):",
             f"{'scenario':16s} {'success b→l':>14s} {'slot RMSE b→l':>20s}"]
    for name, row in comparison["summary"].items():
        s, r = row["success_rate"], row["mean_slot_rmse_m"]
        lines.append(f"{name:16s} {s[0]:>5.2f} → {s[1]:<5.2f} {r[0]:>8.3f} → {r[1]:<8.3f} m")
    return Result("PASS" if passed else "FAIL", lines + notes, path)


def _run_stage(number):
    def run(folder, options):
        from .curriculum import STAGES, train_stage
        checkpoint = train_stage(number, options.seed, retrain=options.retrain)
        return _learned_vs_baseline(folder, options, checkpoint, STAGES[number].scenarios)
    return run


LEARNED_VIEW = "python -m swarmflow view --checkpoint {} --port 8766"

TESTS = (
    Test(1, "unit-checks", "Automated checks (pytest)",
         "Runs every pytest check: PettingZoo API, deterministic seeds, fixed slot identity, swept collision "
         "detection, acceleration limits, sensor masks, the recovery rule, PPO/GAE maths and checkpoint reload.",
         "every pytest check passes", "docs/validation.md · Automated checks", "~1 min", run_unit_checks),
    Test(2, "baseline", "Eight-drone predictive baseline",
         "Flies 8 drones with the predictive controller through 8 scenarios, 3 seeds each (1000-1002), "
         "24 episodes of 24 s. Reproduces the headline result in docs/validation.md.",
         f"no collisions or failures, every crossing event recovered, slot error < {FORMATION_RMSE_M:g} m when "
         "undisturbed",
         "docs/validation.md · Eight-drone baseline", "~2 min",
         _baseline_test(VALIDATION_SCENARIOS, 3, 1000, strict=False),
         VALIDATION_SCENARIOS, {"drones": 8, "seed": 1000}),
    Test(3, "no-avoidance", "No-avoidance control (slot-only)",
         "Same obstacle scenarios, but drones only chase their slots and ignore objects. This is a control "
         "experiment: it shows the obstacles really are on a collision course.",
         "every obstacle episode ends in a collision", "docs/validation.md · Controls", "~30 s",
         run_no_avoidance, OBSTACLE_SCENARIOS, {"drones": 8, "seed": 1000, "controller": "slot"}),
    Test(4, "scaling-16", "16-drone scaling",
         "Doubles the swarm to 16 drones for crossing and head-on objects, seeds 2000-2001, to check the "
         "controller handles a larger team.",
         f"no collisions or failures, every crossing event recovered, slot error < {FORMATION_RMSE_M:g} m when "
         "undisturbed",
         "docs/validation.md · Controls and scaling", "~1 min",
         _baseline_test(("crossing", "head_on"), 2, 2000, {"drones": 16}, strict=False),
         ("crossing", "head_on"), {"drones": 16, "seed": 2000}),
    Test(5, "heldout", "Held-out baseline (seeds 100000+)",
         "Runs the 8 scenarios on 20 untouched seeds starting at 100000 that were never used for tuning, "
         "160 episodes in total.",
         f"no failures, >={RECOVERY_RATE:.0%} of events recovered within {RECOVERY_LIMIT_S:g} s, slot error "
         f"< {FORMATION_RMSE_M:g} m when undisturbed",
         "docs/research-plan.md · acceptance targets; Makefile eval-heldout", "~8 min",
         _baseline_test(VALIDATION_SCENARIOS, 20, 100000),
         VALIDATION_SCENARIOS, {"drones": 8, "seed": 100000}),
    Test(6, "noisy", "Noisy sensing stress test",
         "Adds 5 cm position noise and drops 10% of neighbor/obstacle observations, then repeats the 8 "
         "scenarios with 3 seeds.",
         f"no failures, >={RECOVERY_RATE:.0%} of events recovered within {RECOVERY_LIMIT_S:g} s, slot error "
         f"< {FORMATION_RMSE_M:g} m when undisturbed",
         "Makefile eval-noisy", "~2 min",
         _baseline_test(VALIDATION_SCENARIOS, 3, 1000, {"noise": 0.05, "dropout": 0.1}),
         VALIDATION_SCENARIOS, {"drones": 8, "seed": 1000, "noise": 0.05, "dropout": 0.1}),
    Test(7, "dense", "Dense obstacle stress test",
         "Flies through a cluster of 8-12 static spheres and a slalom course, with 12 obstacle sensor slots, "
         "3 seeds.",
         f"no failures, >={RECOVERY_RATE:.0%} of events recovered within {RECOVERY_LIMIT_S:g} s, slot error "
         f"< {FORMATION_RMSE_M:g} m when undisturbed",
         "Makefile eval-dense", "~30 s",
         _baseline_test(("dense", "slalom"), 3, 1000, {"obstacles": 12}),
         ("dense", "slalom"), {"drones": 8, "seed": 1000, "obstacles": 12}),
    Test(8, "smoke-training", "Learning pipeline smoke test",
         "Trains a tiny neural policy (behavior cloning + 512 PPO steps), saves and reloads the checkpoint, "
         "and evaluates it. Checks the machinery only; the policy is not expected to fly well.",
         "training, checkpoint reload and evaluation all complete",
         "docs/validation.md · Learning pipeline; Makefile train-smoke", "~1 min", run_smoke_training,
         ("follow", "crossing"), watch=LEARNED_VIEW.format("runs/tests/08-smoke-training-seed0/latest.pt")),
    Test(9, "stage1", "Stage 1 training: hold/follow/turn",
         "Evaluates models/stage1-seed0/best.pt (training it first if needed: DAgger behavior cloning, then 1M "
         "PPO steps on hold, follow and turn) against the predictive baseline on the same 20 seeds.",
         f"learned policy: no failures, >={RECOVERY_RATE:.0%} of events recovered within {RECOVERY_LIMIT_S:g} s, "
         f"slot error < {FORMATION_RMSE_M:g} m on hold/follow/turn",
         "training.md · Stage 1; make train-stage1", "reuses model, else ~30-60 min", _run_stage(1),
         ("hold", "follow", "turn"), watch=LEARNED_VIEW.format("models/stage1-seed0/best.pt")),
    Test(10, "stage2", "Stage 2 training: + crossing/head-on",
         "Evaluates models/stage2-seed0/best.pt (training it first if needed: 2M PPO steps from the stage 1 best, "
         "adding crossing and head-on objects) against the baseline.",
         f"learned policy: no failures, >={RECOVERY_RATE:.0%} of events recovered within {RECOVERY_LIMIT_S:g} s, "
         f"slot error < {FORMATION_RMSE_M:g} m on hold/follow/turn",
         "training.md · Stage 2; make train-stage2", "reuses model, else ~1-2 h", _run_stage(2),
         ("hold", "follow", "turn", "crossing", "head_on"), watch=LEARNED_VIEW.format("models/stage2-seed0/best.pt")),
    Test(11, "stage3", "Stage 3 training: full benchmark",
         "Evaluates models/stage3-seed0/best.pt (training it first if needed: 2M PPO steps from the stage 2 best "
         "with repeated crossings and wind) against the baseline on the seven learned-policy scenarios.",
         f"learned policy: no failures, >={RECOVERY_RATE:.0%} of events recovered within {RECOVERY_LIMIT_S:g} s, "
         f"slot error < {FORMATION_RMSE_M:g} m on hold/follow/turn",
         "training.md · Stage 3; make train-stage3", "reuses model, else ~2 h", _run_stage(3),
         ("hold", "follow", "crossing", "head_on", "double_crossing", "turn", "gust"),
         watch=LEARNED_VIEW.format("models/stage3-seed0/best.pt")),
)


def _folder(test, options):
    suffix = f"-seed{options.seed}" if test.number in (8, 9, 10, 11) else ""
    return TESTS_DIR / f"{test.number:02d}-{test.slug}{suffix}"


def catalog():
    """Test metadata plus the last recorded result, as served to the browser viewer."""
    rows = []
    for test in TESTS:
        saved = _folder(test, Options()) / "result.json"
        try:
            last = json.loads(saved.read_text())
        except (OSError, json.JSONDecodeError):
            last = None
        rows.append({"number": test.number, "slug": test.slug, "title": test.title, "purpose": test.purpose,
                     "criterion": test.criterion, "source": test.source, "estimate": test.estimate,
                     "scenarios": list(test.scenarios), "setup": test.setup, "watch": test.watch,
                     "command": f"make test-{test.number}", "last": last})
    return rows


def print_catalog():
    print("SwarmFlow test catalog\n")
    for test in TESTS:
        print(f"  {test.number:>2}. {test.title:<40s} {test.estimate:>11s}")
        print(f"      {test.purpose}")
        print(f"      pass: {test.criterion}")
        print(f"      source: {test.source}")
    print("\nRun with e.g. `python -m swarmflow test 2 3` or `python -m swarmflow test all`.")


def resolve(selection):
    by_key = {str(t.number): t for t in TESTS} | {t.slug: t for t in TESTS}
    chosen = []
    for item in selection:
        for part in item.replace(",", " ").split():
            if part == "all":
                chosen.extend(TESTS)
            elif "-" in part and all(p.isdigit() for p in part.split("-", 1)):
                low, high = map(int, part.split("-", 1))
                chosen.extend(t for t in TESTS if low <= t.number <= high)
            elif part in by_key:
                chosen.append(by_key[part])
            else:
                raise ValueError(f"Unknown test '{part}'; use --list to see the catalog")
    return list(dict.fromkeys(chosen))


def prompt():
    print_catalog()
    try:
        answer = input("\nWhich tests? (numbers like `2 3`, ranges like `2-4`, `all`, or `q` to quit): ").strip()
    except EOFError:
        return []
    return [] if answer.lower() in ("", "q", "quit") else resolve([answer])


def run_tests(selection, options):
    tests = resolve(selection) if selection else prompt()
    if not tests:
        print("No tests selected.")
        return 0
    results = []
    for test in tests:
        folder = _folder(test, options)
        folder.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Test {test.number}: {test.title} ===\n{test.purpose}\nOutput: {folder}\n", flush=True)
        began = time.monotonic()
        try:
            result = test.run(folder, options)
        except (ValueError, FileNotFoundError) as error:
            result = Result("ERROR", [str(error)])
        elapsed = time.monotonic() - began
        print(f"\n--- Test {test.number} result: {result.verdict} ({elapsed:.0f} s) ---")
        for line in result.notes:
            print(f"  {line}")
        if result.report:
            print(f"  report: {result.report}")
        results.append((test, result, elapsed))
        (folder / "result.json").write_text(json.dumps({
            "test": test.number, "verdict": result.verdict, "notes": result.notes, "report": result.report,
            "elapsed_s": round(elapsed, 1), "finished": datetime.now().isoformat(timespec="seconds")}, indent=2) + "\n")
    print("\n=== Summary ===")
    for test, result, elapsed in results:
        print(f"  {test.number:>2}. {test.title:<40s} {result.verdict:<6s} {elapsed:>6.0f} s")
    return 0 if all(r.verdict == "PASS" for _, r, _ in results) else 1
