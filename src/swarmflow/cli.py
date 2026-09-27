import argparse
from dataclasses import replace

from .config import Config
from .scenarios import SCENARIOS


def main():
    parser = argparse.ArgumentParser(description="SwarmFlow: target following and formation recovery")
    commands = parser.add_subparsers(dest="command", required=True)
    view = commands.add_parser("view", help="Open an interactive local flight laboratory")
    view.add_argument("--port", type=int, default=8765)
    view.add_argument("--checkpoint")
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
    training = commands.add_parser("train", help="Train a shared local policy with PPO")
    training.add_argument("--steps", type=int, default=10000, help="Environment steps, each includes every drone")
    training.add_argument("--horizon", type=int, default=128)
    training.add_argument("--epochs", type=int, default=4)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["hold", "follow"])
    training.add_argument("--clone-steps", type=int, default=0)
    training.add_argument("--clone-epochs", type=int, default=15)
    training.add_argument("--resume")
    training.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    training.add_argument("--output", default="runs/train")
    training.add_argument("--critic", choices=("local", "central"), default="local")
    training.add_argument("--failure-reset", action="store_true")
    for command in (view, evaluation, training):
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
        config = Config()
        path = getattr(args, "checkpoint", None) or getattr(args, "resume", None)
        if path:
            from .policy import LearnedController
            config = LearnedController(path).config
        overrides = {key: getattr(args, key) for key in
                     ("drones", "duration", "noise", "dropout", "obstacles", "rays", "ray_range")
                     if getattr(args, key, None) is not None}
        config = replace(config, **overrides)
        if args.command == "view":
            if path and overrides:
                parser.error("For learned viewing, use checkpoint configuration; change swarm size in the viewer")
            from .server import serve
            serve(config, args.port, args.checkpoint)
        elif args.command == "eval":
            if args.seeds < 1 or args.seed_start < 0:
                parser.error("Seed count must be positive and seed start nonnegative")
            if args.policy != "learned" and args.checkpoint:
                parser.error("--checkpoint requires --policy learned")
            from .evaluate import evaluate
            report = evaluate(config, args.scenarios, range(args.seed_start, args.seed_start + args.seeds),
                              args.policy, args.checkpoint, args.output, args.trace)
            print(f"Saved {len(report['episodes'])} episodes to {args.output}")
        else:
            from .train import train
            checkpoint = train(config, args.scenarios, args.steps, args.horizon, args.epochs, args.seed,
                               args.output, args.clone_steps, args.clone_epochs, args.resume, args.device,
                               critic=args.critic, failure_reset=args.failure_reset)
            print(f"Checkpoint: {checkpoint}. Evaluate on held-out seeds before judging performance.")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
