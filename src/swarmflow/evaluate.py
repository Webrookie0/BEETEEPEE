import json
from pathlib import Path
import numpy as np

from .baseline import PredictiveController, SlotController
from .config import Config
from .env import SwarmEnv
from .scenarios import SCENARIOS


def run_episode(config, scenario, seed, controller, trace=False):
    env = SwarmEnv(config, scenario)
    observations, _ = env.reset(seed=seed)
    frames = [env.snapshot()] if trace else None
    while env.agents:
        observations, _, _, _, _ = env.step(controller(observations))
        if trace and (env.steps % 2 == 0 or not env.agents):
            frames.append(env.snapshot())
    summary = env.summary()
    env.close()
    return summary, frames


def aggregate(episodes):
    result = {}
    for scenario in dict.fromkeys(e["scenario"] for e in episodes):
        rows = [e for e in episodes if e["scenario"] == scenario]
        events = [(event, row) for row in rows for event in row["events"]]
        recovered = [event["recovery_s"] for event, row in events
                     if event["recovery_s"] is not None and not row["failure"]]
        result[scenario] = {"episodes": len(rows), "failures": sum(bool(e["failure"]) for e in rows),
            "collisions": sum(e["collision"] for e in rows),
            "mean_slot_rmse_m": float(np.mean([e["slot_rmse_m"] for e in rows])),
            "min_surface_clearance_m": min(e["min_surface_clearance_m"] for e in rows),
            "events": len(events), "recovered_events": len(recovered),
            "mean_recovery_s_conditional": float(np.mean(recovered)) if recovered else None,
            "max_recovery_s_conditional": max(recovered) if recovered else None}
    return result


def evaluate(config=None, scenarios=SCENARIOS, seeds=range(1000, 1003), policy="baseline", checkpoint=None,
             output="runs/evaluation.json", trace=False):
    config = config or Config()
    if policy == "learned":
        from .policy import LearnedController
        if checkpoint is None:
            raise ValueError("--checkpoint is required for learned evaluation")
        controller = LearnedController(checkpoint)
    else:
        controller = PredictiveController(config) if policy == "baseline" else SlotController(config)
    episodes = []
    for scenario in scenarios:
        for seed in seeds:
            summary, frames = run_episode(config, scenario, seed, controller, trace)
            episodes.append(summary)
            if frames is not None:
                target = Path(output).with_name(f"{Path(output).stem}-{scenario}-{seed}-trace.json")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(frames))
            print(f"{scenario:16s} seed={seed} failure={summary['failure']} "
                  f"slot_rmse={summary['slot_rmse_m']:.3f} m", flush=True)
    report = {"model": "3D stabilized flight; bounded acceleration and finite velocity response",
              "policy": policy, "checkpoint": str(checkpoint) if checkpoint else None,
              "config": config.to_dict(), "summary": aggregate(episodes), "episodes": episodes}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
