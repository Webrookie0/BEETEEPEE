"""Training curriculum, the resumable full pipeline, and the showcase (best model) folder."""
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config

MODELS_DIR = Path("models")
SHOWCASE_DIR = MODELS_DIR / "showcase"
TEST_SEED_START = 100000
SHOWCASE_SCENARIOS = ("hold", "follow", "crossing", "head_on", "double_crossing", "slalom", "turn", "gust")


@dataclass(frozen=True)
class Stage:
    number: int
    title: str
    scenarios: tuple
    steps: int
    resume_from: int | None = None
    clone_steps: int = 0
    dagger_rounds: int = 0


STAGES = {
    1: Stage(1, "Formation keeping: hold, follow and turn with the target",
             ("hold", "follow", "turn"), 1_000_000, clone_steps=10_000, dagger_rounds=3),
    2: Stage(2, "Moving obstacles: crossing and head-on objects, keeping the stage 1 tasks",
             ("hold", "follow", "turn", "crossing", "head_on"), 2_000_000, resume_from=1),
    3: Stage(3, "Full benchmark: repeated crossings, static spheres and wind gusts",
             SHOWCASE_SCENARIOS, 2_000_000, resume_from=2),
}

# Shared PPO settings for every stage; 8 environments x 256 steps x 8 drones = 16,384 samples per update.
COMMON = {"envs": 8, "horizon": 256, "epochs": 5, "minibatch": 2048, "lr": 3e-4, "eval_every": 100_000,
          "eval_seeds": 5, "snapshot_every": 500_000}


def stage_dir(number, seed, models_dir=MODELS_DIR):
    return Path(models_dir) / f"stage{number}-seed{seed}"


def stage_arguments(number, seed=0, models_dir=MODELS_DIR):
    """Keyword arguments for train() that reproduce one curriculum stage."""
    stage = STAGES[number]
    args = {**COMMON, "stage": number, "seed": seed, "scenarios": list(stage.scenarios),
            "eval_scenarios": list(stage.scenarios), "total_steps": stage.steps,
            "clone_steps": stage.clone_steps, "dagger_rounds": stage.dagger_rounds,
            "output": str(stage_dir(number, seed, models_dir))}
    if stage.resume_from:
        args["resume"] = str(stage_dir(stage.resume_from, seed, models_dir) / "best.pt")
    return args


def train_stage(number, seed=0, models_dir=MODELS_DIR, config=None, retrain=False, **overrides):
    """Train one stage, or finish it if it was interrupted; skip it if it already finished."""
    from .policy import LearnedController
    from .train import train
    folder = stage_dir(number, seed, models_dir)
    started_without_update = folder.exists() and not (folder / "latest.pt").exists() and any(folder.iterdir())
    if (retrain or started_without_update) and folder.exists():
        from datetime import datetime
        archived = folder.with_name(f"{folder.name}-old-{datetime.now():%Y%m%d-%H%M%S}")
        folder.rename(archived)
        print(f"Moved previous run to {archived}", flush=True)
    if (folder / "finished.json").exists():
        print(f"Stage {number} (seed {seed}) already finished in {folder}; skipping.", flush=True)
        return folder / "best.pt"
    args = {**stage_arguments(number, seed, models_dir), **overrides}
    print(f"\n=== Stage {number}, seed {seed}: {STAGES[number].title} ===", flush=True)
    if (folder / "latest.pt").exists():
        print(f"Continuing the interrupted run in {folder}", flush=True)
        args["continue_run"] = True
        config = LearnedController(folder / "latest.pt").config
    elif args.get("resume"):
        if not Path(args["resume"]).exists():
            raise ValueError(f"Stage {number} starts from {args['resume']}; train stage {number - 1} first")
        config = LearnedController(args["resume"]).config
    train(config or Config(), **args)
    return folder / "best.pt"


def run_pipeline(seeds=(0,), stages=(1, 2, 3), models_dir=MODELS_DIR, config=None, showcase=True,
                 test_seeds=20, **overrides):
    total = sum(STAGES[n].steps for n in stages) * len(seeds)
    print("SwarmFlow training pipeline", flush=True)
    for number in stages:
        stage = STAGES[number]
        print(f"  stage {number}: {stage.steps:,} PPO steps on {', '.join(stage.scenarios)}"
              + (f" (after {stage.dagger_rounds + 1} behavior-cloning rounds)" if stage.clone_steps else ""),
              flush=True)
    print(f"  seeds {list(seeds)}: {total:,} PPO steps in total, results in {Path(models_dir)}/stageN-seedS/; "
          f"the best model goes to {Path(models_dir) / 'showcase'}/", flush=True)
    print("  Re-run the same command after an interruption to continue where it stopped.", flush=True)
    for seed in seeds:
        for number in stages:
            train_stage(number, seed, models_dir, config, **overrides)
    if showcase:
        build_showcase(trained_runs(models_dir, max(stages)), Path(models_dir) / "showcase", test_seeds)


def trained_runs(models_dir=MODELS_DIR, stage=None):
    """Run folders (one per seed) that already have a best.pt, for `stage` or else the highest stage."""
    found = {}
    for folder in Path(models_dir).glob("stage*-seed*"):
        match = re.fullmatch(r"stage(\d+)-seed(\d+)", folder.name)
        if match and (folder / "best.json").exists():
            found.setdefault(int(match.group(1)), []).append(folder)
    if not found:
        return []
    return sorted(found.get(stage, []) if stage is not None else found[max(found)])


def training_chain(run):
    """Run folders from the first stage to `run`, following each stage's initial checkpoint."""
    chain, folder = [], Path(run)
    while folder is not None and (folder / "config.json").exists() and folder not in chain:
        chain.append(folder)
        settings = json.loads((folder / "config.json").read_text())
        source = settings.get("initialized_from") or settings.get("resume")
        folder = Path(source).parent if source else None
    return chain[::-1]


def _stage_label(folder):
    try:
        settings = json.loads((Path(folder) / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return Path(folder).name
    return f"Stage {settings['stage']}" if settings.get("stage") else Path(folder).name


def build_showcase(runs=None, output=SHOWCASE_DIR, test_seeds=20, seed_start=TEST_SEED_START,
                   scenarios=SHOWCASE_SCENARIOS):
    """Select the best trained run, test it on held-out seeds against the baseline and store it.

    The candidate is chosen by its validation (selection-seed) score only; the held-out test seeds
    are used once, after the choice, so the reported test result is not tuned to them.
    """
    from .evaluate import compare_reports, evaluate, learning_checks, selection_score
    from .policy import LearnedController
    runs = [Path(r) for r in runs] if runs else trained_runs(Path(output).parent)
    if not runs:
        raise ValueError("No trained runs found; train first (make training)")
    candidates = []
    for run in runs:
        best = run / "best.json"
        if not best.exists():
            raise ValueError(f"{run} has no best.json; train or finish it first")
        row = json.loads(best.read_text())
        candidates.append({"run": str(run), "score": row["score"], "environment_steps": row["environment_steps"],
                           "scenarios": row["scenarios"]})
    if len({tuple(c["scenarios"]) for c in candidates}) > 1:
        raise ValueError("Candidate runs were validated on different scenarios; compare runs of the same stage")
    chosen = max(candidates, key=lambda c: c["score"])
    source = Path(chosen["run"])
    output = Path(output)
    if (output / "best.pt").exists():
        from datetime import datetime
        archive = output.parent / f"{output.name}-archive" / f"{datetime.now():%Y%m%d-%H%M%S}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output), str(archive))
        print(f"Previous showcase kept in {archive}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "best.pt", output / "best.pt")
    shutil.copy2(source / "best.json", output / "validation.json")
    chain = training_chain(source)
    history = output / "history"
    if history.exists():
        shutil.rmtree(history)
    for folder in chain:
        for name in ("config.json", "training.jsonl", "evaluations.jsonl", "best.json", "finished.json"):
            if (folder / name).exists():
                (history / folder.name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(folder / name, history / folder.name / name)
    print(f"Selected {source} (validation score {chosen['score']:.3f} at "
          f"{chosen['environment_steps']:,} steps) from {len(candidates)} run(s)", flush=True)
    checkpoint = str(output / "best.pt")
    config = LearnedController(checkpoint).config
    seeds = range(seed_start, seed_start + test_seeds)
    print(f"Held-out test: {len(scenarios)} scenarios x {test_seeds} seeds ({seeds.start}..{seeds.stop - 1})",
          flush=True)
    learned = evaluate(config, list(scenarios), seeds, "learned", checkpoint, str(output / "test_learned.json"))
    baseline = evaluate(config, list(scenarios), seeds, "baseline", None, str(output / "test_baseline.json"))
    comparison = compare_reports((output / "test_baseline.json", output / "test_learned.json"),
                                 str(output / "test_compare.json"))
    checks = learning_checks(learned)
    table = []
    for name in scenarios:
        b, l = baseline["summary"][name], learned["summary"][name]
        table.append({"scenario": name,
                      "baseline": {k: b[k] for k in ("success_rate", "mean_slot_rmse_m", "events",
                                                      "recovered_events", "min_surface_clearance_m")},
                      "learned": {k: l[k] for k in ("success_rate", "mean_slot_rmse_m", "events",
                                                     "recovered_events", "min_surface_clearance_m")}})
    summary = {"selected_run": str(source), "selected_environment_steps": chosen["environment_steps"],
               "validation_score": chosen["score"], "candidates": candidates,
               "chain": [folder.name for folder in chain], "stages": [_stage_label(f) for f in chain],
               "training_stages": [_stage_record(folder) for folder in chain],
               "test_seeds": [seeds.start, seeds.stop - 1], "test_scenarios": list(scenarios),
               "test_score_learned": selection_score(learned["summary"]),
               "test_score_baseline": selection_score(baseline["summary"]),
               "learned_correctly": all(check["passed"] for check in checks),
               "checks": checks, "scenarios": table, "comparison": comparison["summary"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "SUMMARY.md").write_text(_summary_markdown(summary))
    print(f"\nShowcase saved in {output}. Learned correctly: {'YES' if summary['learned_correctly'] else 'NOT YET'}",
          flush=True)
    for check in checks:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['name']}: {check['detail']}", flush=True)
    return summary


def _stage_record(folder):
    def read(name):
        try:
            return json.loads((Path(folder) / name).read_text())
        except (OSError, json.JSONDecodeError):
            return {}
    settings, best, finished = read("config.json"), read("best.json"), read("finished.json")
    return {"run": Path(folder).name, "label": _stage_label(folder), "scenarios": settings.get("scenarios"),
            "clone_steps": settings.get("clone_steps"), "dagger_rounds": settings.get("dagger_rounds"),
            "environment_steps": finished.get("environment_steps"), "finished": bool(finished),
            "best_score": best.get("score"), "best_environment_steps": best.get("environment_steps"),
            "training_seconds": finished.get("training_seconds_this_run")}


def _summary_markdown(summary):
    def cell(row, key):
        value = row[key]
        return "-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)

    lines = ["# Showcase model", "",
             f"- Model: `best.pt`, copied from `{summary['selected_run']}/best.pt`",
             f"- Trained through: {' -> '.join(summary['chain'])}",
             f"- Chosen at {summary['selected_environment_steps']:,} environment steps with validation score "
             f"{summary['validation_score']:.3f} (selection seeds 5000+)",
             f"- Held-out test seeds: {summary['test_seeds'][0]}..{summary['test_seeds'][1]}; "
             f"test score learned {summary['test_score_learned']:.3f} vs baseline {summary['test_score_baseline']:.3f}",
             f"- **Learned correctly: {'YES' if summary['learned_correctly'] else 'NOT YET'}**", "",
             "| Stage | Scenarios | Steps at end | Best validation score | Best at step |",
             "| --- | --- | --- | --- | --- |"]
    for stage in summary.get("training_stages", []):
        steps = "-" if stage["environment_steps"] is None else f"{stage['environment_steps']:,}"
        best = "-" if stage["best_score"] is None else f"{stage['best_score']:.3f}"
        at = "-" if stage["best_environment_steps"] is None else f"{stage['best_environment_steps']:,}"
        lines.append(f"| {stage['label']} | {', '.join(stage['scenarios'] or [])} | {steps} | {best} | {at} |")
    lines += ["", "| Check | Result | Detail |", "| --- | --- | --- |"]
    lines += [f"| {c['name']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |" for c in summary["checks"]]
    lines += ["", "| Scenario | Success (baseline / learned) | Slot RMSE m (baseline / learned) | "
              "Recovered events (baseline / learned) |", "| --- | --- | --- | --- |"]
    for row in summary["scenarios"]:
        b, l = row["baseline"], row["learned"]
        lines.append(f"| {row['scenario']} | {cell(b, 'success_rate')} / {cell(l, 'success_rate')} | "
                     f"{cell(b, 'mean_slot_rmse_m')} / {cell(l, 'mean_slot_rmse_m')} | "
                     f"{b['recovered_events']}/{b['events']} / {l['recovered_events']}/{l['events']} |")
    lines += ["", "View it: `python -m swarmflow view` (loads this model automatically).", ""]
    return "\n".join(lines)


def default_model(models_dir=MODELS_DIR):
    """The checkpoint the viewer should show: the showcase model, else the best stage checkpoint."""
    showcase = Path(models_dir) / "showcase" / "best.pt"
    if showcase.exists():
        return showcase
    runs = trained_runs(models_dir)
    if not runs:
        return None
    best = max(runs, key=lambda run: json.loads((run / "best.json").read_text())["score"])
    return best / "best.pt" if (best / "best.pt").exists() else None


def _read_jsonl(path):
    rows = []
    try:
        with open(path) as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        pass
    return rows


def _thin(rows, limit=300):
    if len(rows) <= limit:
        return rows
    step = len(rows) / limit
    return [rows[int(i * step)] for i in range(limit)] + [rows[-1]]


def _history_folders(checkpoint):
    folder = Path(checkpoint).parent
    summary = folder / "summary.json"
    if summary.exists():
        names = json.loads(summary.read_text()).get("chain", [])
        return [folder / "history" / name for name in names if (folder / "history" / name).exists()]
    return training_chain(folder)


def model_report(checkpoint):
    """What the viewer shows about a trained model: training history, selection and held-out test."""
    from .policy import LearnedController
    checkpoint = Path(checkpoint)
    metadata = LearnedController(checkpoint).metadata
    training = metadata.get("training") or {}
    stages = []
    # Each stage restarts from the previous stage's best checkpoint, whose step count can be lower than
    # where that stage stopped; `x` lays the stages end to end so the curves never overlap.
    pipeline_steps = 0
    for folder in _history_folders(checkpoint):
        settings = {}
        if (folder / "config.json").exists():
            settings = json.loads((folder / "config.json").read_text())
        evaluations = [{k: row.get(k) for k in ("environment_steps", "score", "best", "success_rate",
                                                 "mean_slot_rmse_m", "events", "recovered_events", "reason")}
                       for row in _read_jsonl(folder / "evaluations.jsonl")]
        updates = [{k: row.get(k) for k in ("environment_steps", "phase", "mean_step_reward",
                                             "recent_failure_rate", "recent_slot_rmse_m", "action_std",
                                             "environment_steps_per_second")}
                   for row in _read_jsonl(folder / "training.jsonl") if row.get("type") == "ppo"]
        cloning = [row for row in _read_jsonl(folder / "training.jsonl") if row.get("type") == "behavior_cloning"]
        start = settings.get("stage_start_environment_steps") or 0
        for row in evaluations + updates:
            row["x"] = pipeline_steps + row["environment_steps"] - start
        end = max((row["environment_steps"] for row in evaluations + updates), default=start)
        stages.append({"name": folder.name, "label": _stage_label(folder), "stage": settings.get("stage"),
                       "scenarios": settings.get("scenarios"), "algorithm": settings.get("algorithm"),
                       "target_environment_steps": settings.get("target_environment_steps"),
                       "ppo_steps": end - start, "finished": (folder / "finished.json").exists(),
                       "cloning": cloning, "evaluations": evaluations, "updates": _thin(updates)})
        pipeline_steps += end - start
    report = {"checkpoint": str(checkpoint), "format": metadata.get("format"), "status": metadata.get("status"),
              "environment_steps": metadata.get("environment_steps"), "pipeline_steps": pipeline_steps,
              "drones": metadata["config"]["drones"],
              "algorithm": training.get("algorithm"),
              "scenarios": training.get("scenarios"), "stage": training.get("stage"),
              "selection": metadata.get("selection"), "stages": stages, "test": None}
    summary = checkpoint.parent / "summary.json"
    if summary.exists():
        report["test"] = json.loads(summary.read_text())
    return report


def training_status(models_dir=MODELS_DIR, active_within_s=900):
    """Progress of the most recently active stage run, shown live in the viewer while training runs."""
    runs = [folder for folder in Path(models_dir).glob("stage*-seed*")
            if re.fullmatch(r"stage\d+-seed\d+", folder.name) and (folder / "config.json").exists()]
    if not runs:
        return None

    def touched(folder):
        return max((p.stat().st_mtime for p in folder.iterdir() if p.is_file()), default=0.0)

    run = max(runs, key=touched)
    try:
        settings = json.loads((run / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    rows = _read_jsonl(run / "training.jsonl")
    updates = [row for row in rows if row.get("type") == "ppo"]
    evaluations = _read_jsonl(run / "evaluations.jsonl")
    start = settings.get("stage_start_environment_steps") or 0
    target = settings.get("target_environment_steps")
    last = updates[-1] if updates else {}
    steps = last.get("environment_steps", start)
    rate = last.get("environment_steps_per_second")
    finished = (run / "finished.json").exists()
    age = time.time() - touched(run)
    if finished:
        phase = "finished"
    elif last:
        phase = last.get("phase", "ppo")
    else:
        phase = "starting PPO" if evaluations else "behavior cloning"
    return {"run": str(run), "stage": settings.get("stage"), "stages": len(STAGES),
            "scenarios": settings.get("scenarios"), "phase": phase, "finished": finished,
            "active": not finished and age < active_within_s, "seconds_since_update": round(age),
            "environment_steps": steps, "start_environment_steps": start, "target_environment_steps": target,
            "progress": (steps - start) / (target - start) if target and target > start else None,
            "steps_per_second": rate,
            "eta_seconds": (target - steps) / rate if rate and target and not finished else None,
            "cloning_rounds_done": sum(1 for row in rows if row.get("type") == "behavior_cloning"),
            "recent_failure_rate": last.get("recent_failure_rate"),
            "recent_slot_rmse_m": last.get("recent_slot_rmse_m"),
            "last_score": evaluations[-1].get("score") if evaluations else None,
            "best_score": max((row["score"] for row in evaluations), default=None)}


def start_viewer(port=8765, browser=True, log=Path("runs") / "viewer-training.log"):
    """Start the viewer in its own process so it can show training live without slowing it down."""
    import socket
    import subprocess
    import sys
    import webbrowser
    url = f"http://127.0.0.1:{port}/"

    def listening():
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.3).close()
            return True
        except OSError:
            return False

    if listening():
        print(f"Port {port} is already in use; if a SwarmFlow viewer runs there it follows this training too: "
              f"{url}", flush=True)
        return None
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as handle:
        process = subprocess.Popen([sys.executable, "-m", "swarmflow", "view", "--port", str(port)],
                                   stdout=handle, stderr=subprocess.STDOUT)
    for _ in range(150):
        if process.poll() is not None:
            print(f"The viewer could not start (see {log}); training continues without it.", flush=True)
            return None
        if listening():
            break
        time.sleep(0.1)
    print(f"Viewer: {url}  (live training progress; it switches to each new best model automatically)",
          flush=True)
    if browser:
        webbrowser.open(url)
    return process
