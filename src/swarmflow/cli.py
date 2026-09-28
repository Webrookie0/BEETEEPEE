import argparse
from dataclasses import replace
from pathlib import Path

from .config import Config
from .scenarios import SCENARIOS

TRAIN_OPTIONS = ("total_steps", "horizon", "epochs", "seed", "scenarios", "clone_steps", "clone_epochs",
                 "dagger_rounds", "clone_noise", "resume", "device", "output", "critic", "separate_critic",
                 "failure_reset", "envs", "lr", "minibatch", "entropy", "target_kl", "init_log_std",
                 "critic_warmup", "eval_every", "eval_seeds", "eval_scenarios", "snapshot_every", "threads",
                 "continue_run", "width", "min_log_std", "bc_coef", "bc_coef_end", "teacher_margins", "hard_weight",
                 "gap_features")


def main():
    parser = argparse.ArgumentParser(description="SwarmFlow: target following and formation recovery")
    commands = parser.add_subparsers(dest="command", required=True)
    view = commands.add_parser("view", help="Open an interactive local flight laboratory")
    view.add_argument("--port", type=int, default=8765)
    view.add_argument("--checkpoint", help="Trained policy to load (default: models/showcase/best.pt, "
                                           "else the best stage model under models/)")
    view.add_argument("--baseline-only", action="store_true", help="Do not load any trained policy")
    evaluation = commands.add_parser("eval", help="Run seeded benchmark scenarios")
    evaluation.add_argument("--policy", choices=("baseline", "slot", "learned"), default="baseline")
    evaluation.add_argument("--checkpoint")
    evaluation.add_argument("--seeds", type=int, default=3, help="Number of episodes per scenario")
    evaluation.add_argument("--seed-start", type=int, default=1000)
    evaluation.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    evaluation.add_argument("--output", default="runs/evaluation.json")
    evaluation.add_argument("--trace", action="store_true")
    evaluation.add_argument("--obstacles", type=int)
    evaluation.add_argument("--rays", type=int)
    evaluation.add_argument("--ray-range", type=float, dest="ray_range")
    compare = commands.add_parser("compare", help="Diff success rate and slot RMSE from two reports")
    compare.add_argument("report1")
    compare.add_argument("report2")
    compare.add_argument("--output", default="runs/compare.json")
    suite = commands.add_parser("test", help="Pick and run numbered validation tests (interactive if none given)")
    suite.add_argument("tests", nargs="*", help="Test numbers, ranges or names, e.g. `2 3`, `2-4`, `baseline`, `all`")
    suite.add_argument("--list", action="store_true", help="Show the test catalog and exit")
    suite.add_argument("--episodes", type=int, help="Override the number of seeds per scenario")
    suite.add_argument("--seed", type=int, default=0, help="Training seed for tests 8-11")
    suite.add_argument("--retrain", action="store_true", help="Train again even if a checkpoint exists")
    suite.add_argument("-k", dest="keyword", help="Only run pytest checks matching this keyword (test 1)")

    training = commands.add_parser("train", help="Train the shared drone policy (behavior cloning + PPO)")
    training.add_argument("--stage", type=int, choices=(1, 2, 3),
                          help="Use the curriculum preset for this stage (see training.md); other flags override it")
    training.add_argument("--continue", dest="continue_run", action="store_true", default=None,
                          help="Finish the interrupted run in --output from its latest.pt, with its own settings")
    training.add_argument("--steps", dest="total_steps", type=int,
                          help="Environment steps in this run, summed over the parallel environments")
    training.add_argument("--horizon", type=int, help="Rollout steps per environment before each PPO update")
    training.add_argument("--epochs", type=int, help="PPO passes over each rollout (stopped early by --target-kl)")
    training.add_argument("--seed", type=int)
    training.add_argument("--scenarios", nargs="+", choices=SCENARIOS)
    training.add_argument("--clone-steps", type=int,
                          help="Environment steps per behavior-cloning round; 0 means PPO from random weights")
    training.add_argument("--clone-epochs", type=int)
    training.add_argument("--dagger-rounds", type=int, help="Extra cloning rounds flown by the student itself")
    training.add_argument("--clone-noise", type=float, help="Action noise injected while the teacher flies")
    training.add_argument("--resume", help="Start from this checkpoint (weights, optimizer, reward scale)")
    training.add_argument("--device", choices=("cpu", "mps", "cuda"))
    training.add_argument("--output")
    training.add_argument("--critic", choices=("local", "central"))
    training.add_argument("--shared-critic", dest="separate_critic", action="store_false", default=None,
                          help="Let the critic share the actor encoder (the old, unstable setup)")
    training.add_argument("--failure-reset", action="store_true", default=None)
    training.add_argument("--envs", type=int, help="Parallel environments per rollout")
    training.add_argument("--lr", type=float, help="Adam learning rate, annealed linearly to 10%%")
    training.add_argument("--minibatch", type=int)
    training.add_argument("--entropy", type=float)
    training.add_argument("--target-kl", type=float)
    training.add_argument("--init-log-std", type=float)
    training.add_argument("--min-log-std", type=float, help="Floor on the exploration noise (log standard deviation)")
    training.add_argument("--width", type=int, help="Network width (64 = the original network; stages use 128)")
    training.add_argument("--bc-coef", type=float, help="Imitation-loss weight at the start of the run")
    training.add_argument("--bc-coef-end", type=float, help="Imitation-loss weight at the end of the run")
    training.add_argument("--teacher-margins", type=float, nargs=2, metavar=("SOFT", "HARD"),
                          help="Teacher clearance margins in m while cloning (default 0.6 0.25; baseline uses 0.35 0.08)")
    training.add_argument("--hard-weight", type=float,
                          help="How much more often near-object states are sampled for imitation (default 1: off)")
    training.add_argument("--gap-features", dest="gap_features", action="store_true", default=None,
                          help="Give the network each neighbour's and object's surface gap and closing speed")
    training.add_argument("--critic-warmup", type=int, help="First PPO updates that train only the critic")
    training.add_argument("--eval-every", type=int, help="Validate (and maybe save best.pt) every N environment steps")
    training.add_argument("--eval-seeds", type=int, help="Selection episodes per scenario at each validation")
    training.add_argument("--eval-scenarios", nargs="+", choices=SCENARIOS)
    training.add_argument("--snapshot-every", type=int, help="Keep checkpoints/step_N.pt every N environment steps")
    training.add_argument("--threads", type=int, help="CPU threads for PyTorch (default: up to 4)")

    pipeline = commands.add_parser("pipeline", help="Run the full curriculum (stages 1-3) and build the showcase; "
                                                    "re-run the same command to continue after an interruption")
    pipeline.add_argument("--seeds", type=int, nargs="+", default=[0], help="Independent training seeds")
    pipeline.add_argument("--stages", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3])
    pipeline.add_argument("--models", default="models", help="Folder that receives stageN-seedS and showcase")
    pipeline.add_argument("--test-seeds", type=int, default=20, help="Held-out test episodes per scenario")
    pipeline.add_argument("--no-showcase", action="store_true")
    pipeline.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    pipeline.add_argument("--threads", type=int)
    pipeline.add_argument("--view", action="store_true",
                          help="Also open the viewer: live training progress, then the best model")
    pipeline.add_argument("--port", type=int, default=8765, help="Viewer port for --view")
    pipeline.add_argument("--no-browser", action="store_true", help="With --view, do not open a browser window")

    showcase = commands.add_parser("showcase", help="Pick the best trained run, test it on held-out seeds against "
                                                    "the baseline and store it in models/showcase")
    showcase.add_argument("runs", nargs="*", help="Run folders to choose from (default: every seed of the "
                                                  "highest trained stage)")
    showcase.add_argument("--output", default="models/showcase")
    showcase.add_argument("--seeds", type=int, default=20, help="Held-out test episodes per scenario")
    showcase.add_argument("--seed-start", type=int, default=100000)

    for command in (view, evaluation, training, pipeline):
        command.add_argument("--drones", type=int)
        command.add_argument("--duration", type=float)
        command.add_argument("--noise", type=float)
        command.add_argument("--dropout", type=float)
    args = parser.parse_args()
    try:
        if args.command == "compare":
            from .evaluate import compare_reports
            compare_reports((args.report1, args.report2), args.output)
            print(f"Saved comparison to {args.output}")
            return
        if args.command == "test":
            from .suite import Options, print_catalog, run_tests
            if args.list:
                print_catalog()
                return
            if args.episodes is not None and args.episodes < 1:
                parser.error("--episodes must be positive")
            raise SystemExit(run_tests(args.tests, Options(args.episodes, args.seed, args.retrain, args.keyword)))
        if args.command == "showcase":
            from .curriculum import build_showcase
            if args.seeds < 1:
                parser.error("--seeds must be positive")
            build_showcase(args.runs or None, args.output, args.seeds, args.seed_start)
            return
        overrides = {key: getattr(args, key) for key in
                     ("drones", "duration", "noise", "dropout", "obstacles", "rays", "ray_range")
                     if getattr(args, key, None) is not None}
        if args.command == "pipeline":
            from .curriculum import run_pipeline, start_viewer
            extra = {"device": args.device} | ({"threads": args.threads} if args.threads else {})
            viewer = start_viewer(args.port, not args.no_browser) if args.view else None
            try:
                run_pipeline(args.seeds, sorted(set(args.stages)), args.models,
                             replace(Config(), **overrides) if overrides else None,
                             not args.no_showcase, args.test_seeds, **extra)
                if viewer is not None and viewer.poll() is None:
                    print(f"\nTraining finished. The viewer at http://127.0.0.1:{args.port} now flies the best model "
                          "(models/showcase/best.pt). Press Ctrl+C to stop it.", flush=True)
                    viewer.wait()
            finally:
                if viewer is not None and viewer.poll() is None:
                    viewer.terminate()
            return
        if args.command == "train":
            train_command(args, overrides)
            return
        from .policy import LearnedController
        path = args.checkpoint
        if args.command == "view" and not path and not args.baseline_only:
            from .curriculum import default_model
            found = default_model()
            path = str(found) if found else None
            print(f"Trained policy: {path}" if path else
                  "No trained policy found under models/; showing the predictive baseline only", flush=True)
        config = Config()
        if path and not (args.command == "view" and overrides and not args.checkpoint):
            config = LearnedController(path).config
        config = replace(config, **overrides)
        if args.command == "view":
            if args.checkpoint and overrides:
                parser.error("For learned viewing, use checkpoint configuration; change swarm size in the viewer")
            from .server import serve
            serve(config, args.port, None if args.baseline_only else path,
                  follow=not args.checkpoint and not args.baseline_only)
        else:
            if args.seeds < 1 or args.seed_start < 0:
                parser.error("Seed count must be positive and seed start nonnegative")
            if args.policy != "learned" and args.checkpoint:
                parser.error("--checkpoint requires --policy learned")
            from .evaluate import evaluate
            report = evaluate(config, args.scenarios, range(args.seed_start, args.seed_start + args.seeds),
                              args.policy, args.checkpoint, args.output, args.trace)
            print(f"Saved {len(report['episodes'])} episodes to {args.output}")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        raise SystemExit(130)


def train_command(args, overrides):
    from .policy import LearnedController
    from .train import train
    options = {}
    if args.stage:
        from .curriculum import stage_arguments
        options = stage_arguments(args.stage, args.seed if args.seed is not None else 0)
    options.update({key: getattr(args, key) for key in TRAIN_OPTIONS if getattr(args, key, None) is not None})
    options.setdefault("scenarios", ["hold", "follow"])
    options.setdefault("output", "runs/train")
    if options.get("continue_run"):
        source = Path(options["output"]) / "latest.pt"
        options.pop("resume", None)
    else:
        source = options.get("resume")
    config = LearnedController(source).config if source else Config()
    config = replace(config, **overrides)
    scenarios = options.pop("scenarios")
    latest = train(config, scenarios, **options)
    best = latest.with_name("best.pt")
    print(f"Latest checkpoint: {latest}\nBest validated checkpoint: {best}\n"
          f"Test it on held-out seeds with `python -m swarmflow showcase {latest.parent}`.")
