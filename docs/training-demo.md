# Showing how SwarmFlow learns

> **Updated pipeline.** The full training is now `make training`, or `make run train` to watch it live (see [guide.md](../guide.md)), and the method is explained in [training.md](../training.md). The viewer loads `models/showcase/best.pt` automatically and shows the training curves and a held-out verdict. The short commands below still work for a quick live demonstration, but the sections about the launcher and the saved smoke policy describe the older setup.

## What is saved today

The regular **Launch SwarmFlow.command** runs the programmed predictive controller. Pressing Start flight or Repeat flights simulates that controller; it does not train a neural network.

**To reproduce the earlier working demonstration, use that launcher and check that Active controller says `predictive baseline`.** The `view --checkpoint ...` commands below deliberately open the experimental neural policy instead. Running locally is not the cause of the difference in flight quality. The launcher skips learned-policy servers when looking for an existing demo to reuse.

An experimental neural policy is saved in `runs/smoke-policy/latest.pt`, with its settings in `config.json`, training history in `training.jsonl`, and measured results in `evaluation.json` in the same folder. It received 128 environment steps of examples from the predictive controller, three behavior-cloning epochs, and 512 PPO environment steps. This checks the training and saving machinery. It still drifts and collides in its recorded evaluation, so it is not a successful trained swarm policy.

The saved checkpoint includes neural-network weights, optimizer state, environment configuration, and the PPO step count. The trainer replaces `latest.pt` after each completed PPO update; it does not retain every earlier version. Separate output folders preserve stages. These artifacts are local: `runs/` is ignored by Git, so copy that folder separately when backing up or sharing trained models.

## Explain the learning process

Each drone observes its target-relative state and visible neighbors and obstacles. A shared neural network chooses a velocity command for each drone. The environment advances the simulated flight and returns rewards. PPO uses collected experience to update the network, then repeats the process. The current implementation is parameter-sharing independent PPO with a local critic.

Training changes the network weights. Viewing a checkpoint uses those saved weights to choose actions without updating them. Optional behavior cloning first teaches the network to imitate the predictive controller; the commands below omit cloning to demonstrate PPO from scratch.

## 1. Run a short training demonstration

Open Terminal in the `neww` folder. On this Mac:

```sh
cd /Users/sumit/Documents/Codex/developer/neww
.venv/bin/python -m swarmflow train --steps 10000 --scenarios hold follow --seed 0 --output runs/training-demo/stage1
```

The terminal prints a row after every PPO update. Show `environment_steps`, `mean_step_reward`, and the loss values while the run progresses. `collisions_this_run` is a cumulative count; it cannot decrease. Neither a lower training loss nor one better reward row proves good flight performance.

The output folder receives `latest.pt`, `training.jsonl`, and `config.json`. This is a demonstration budget, not a promise of convergence. A fresh run refuses to overwrite an existing checkpoint; choose a new output folder for another independent run. If interrupted, the last completed PPO update remains saved, provided at least one update finished.

## 2. Measure the saved policy

After training finishes, evaluate it on the same scenarios using seeds outside the training range of 0–999:

```sh
.venv/bin/python -m swarmflow eval --policy learned --checkpoint runs/training-demo/stage1/latest.pt --scenarios hold follow --seeds 5 --seed-start 3000 --output runs/training-demo/stage1/evaluation.json
.venv/bin/python -m swarmflow eval --policy baseline --scenarios hold follow --seeds 5 --seed-start 3000 --output runs/training-demo/baseline-evaluation.json
```

Compare failures/collisions, formation-slot error, and surface clearance in the reports. When adding obstacle scenarios, also compare event recovery. Use the same scenarios, seeds, drone count, and duration for comparisons. These small repeated demonstration checks are development validation, not an untouched final test set.

## 3. Show the learned flight in the browser

```sh
.venv/bin/python -m swarmflow view --checkpoint runs/training-demo/stage1/latest.pt --port 8766
```

Open **http://127.0.0.1:8766**, choose **hold** or **follow**, and start a flight. The policy indicator should say **learned checkpoint**. Port 8766 lets you keep the regular baseline demo on 8765 for comparison. The learned flight may still fail; show the actual result.

This viewer loads a fixed checkpoint at startup. It does not display live training or reload weights automatically. Stop this viewer with Control+C and restart it with the desired checkpoint to show a later stage. Use the regular launcher when presenting the working predictive baseline.

To inspect the already saved smoke policy instead:

```sh
.venv/bin/python -m swarmflow view --checkpoint runs/smoke-policy/latest.pt --port 8766
```

Run only one viewer on port 8766 at a time.

## 4. Continue training and preserve the first stage

```sh
.venv/bin/python -m swarmflow train --resume runs/training-demo/stage1/latest.pt --steps 40000 --scenarios hold follow --seed 0 --output runs/training-demo/stage2
.venv/bin/python -m swarmflow eval --policy learned --checkpoint runs/training-demo/stage2/latest.pt --scenarios hold follow --seeds 5 --seed-start 3000 --output runs/training-demo/stage2/evaluation.json
```

`--steps 40000` means 40,000 additional PPO environment steps. After a completed 10,000-step first stage, the second checkpoint records 50,000 total PPO environment steps. Resume restores weights and optimizer state but starts a fresh seeded episode; it does not restore an exact point in the previous flight. Compare both saved evaluations and reopen the viewer with `stage2/latest.pt`.

Build reliable holding and following first, then introduce crossing obstacles while retaining those simpler tasks. Progress only when evaluation supports it. Longer training alone does not guarantee improvement; a strong final controller still needs curriculum experiments, tuning, multiple independent training seeds, and broader validation.

## A clear presentation description

“This is a drone-swarm learning environment. The working demo uses a predictive controller as our reference. A separate PPO policy learns inside the same environment. Here are its training updates, saved checkpoints, and evaluations, followed by its actual flight behavior.”
