import json
import threading
import time
from urllib.request import ProxyHandler, Request, build_opener

from swarmflow import Config
from swarmflow.curriculum import (build_showcase, default_model, model_report, stage_dir, trained_runs,
                                  training_status)
from swarmflow.server import make_server
from swarmflow.train import train

urlopen = build_opener(ProxyHandler({})).open


def _tiny_run(models, seed):
    output = stage_dir(1, seed, models)
    train(Config(drones=4, duration=.2), ["hold"], total_steps=8, horizon=8, epochs=1, output=output, envs=1,
          eval_seeds=1, seed=seed, stage=1)
    return output


def test_showcase_selects_best_run_and_viewer_reports_it(tmp_path):
    models = tmp_path / "models"
    runs = [_tiny_run(models, 0), _tiny_run(models, 1)]
    assert sorted(trained_runs(models)) == sorted(runs)
    summary = build_showcase(None, models / "showcase", test_seeds=1, scenarios=("hold", "follow"))
    scores = {str(run): json.loads((run / "best.json").read_text())["score"] for run in runs}
    assert summary["validation_score"] == max(scores.values())
    assert summary["selected_run"] in scores
    assert summary["test_seeds"] == [100000, 100000]
    assert [row["scenario"] for row in summary["scenarios"]] == ["hold", "follow"]
    assert isinstance(summary["learned_correctly"], bool)
    assert default_model(models) == models / "showcase" / "best.pt"
    assert (models / "showcase" / "SUMMARY.md").exists()

    report = model_report(models / "showcase" / "best.pt")
    assert report["test"]["selected_run"] == summary["selected_run"]
    assert [stage["label"] for stage in report["stages"]] == ["Stage 1"]
    assert [e["reason"] for e in report["stages"][0]["evaluations"]] == ["start", "final"]
    assert report["pipeline_steps"] == 8
    assert summary["training_stages"][0]["best_score"] == summary["validation_score"]

    status = training_status(models)
    assert status["phase"] == "finished" and not status["active"] and status["progress"] == 1.0

    build_showcase(None, models / "showcase", test_seeds=1, scenarios=("hold",))
    archived = list((models / "showcase-archive").glob("*/best.pt"))
    assert len(archived) == 1 and (models / "showcase" / "best.pt").exists()

    server = make_server(Config(), 0, str(models / "showcase" / "best.pt"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        state = json.load(urlopen(f"{base}/api/state"))
        assert state["mode"] == "learned" and state["learned_available"]
        assert json.load(urlopen(f"{base}/api/model"))["test"]["checks"]
        request = Request(f"{base}/api/reset", data=json.dumps({"controller": "baseline"}).encode(),
                          headers={"Content-Type": "application/json"})
        assert json.load(urlopen(request))["policy"] == "predictive baseline"
    finally:
        server.shutdown()
        server.server_close()


def test_viewer_follows_new_best_model_during_training(tmp_path):
    models = tmp_path / "models"
    server = make_server(Config(drones=4, duration=.2), 0, None, follow=True, models_dir=models, interval=.05)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        state = json.load(urlopen(f"{base}/api/state"))
        assert state["mode"] == "baseline" and not state["learned_available"]
        assert json.load(urlopen(f"{base}/api/model")) == {"available": False, "training": None}
        run = _tiny_run(models, 0)
        deadline = time.time() + 10
        while not state["learned_available"] and time.time() < deadline:
            time.sleep(.1)
            state = json.load(urlopen(f"{base}/api/state"))
        assert state["mode"] == "learned" and state["checkpoint"] == str(run / "best.pt")
        model = json.load(urlopen(f"{base}/api/model"))
        assert model["available"] and model["training"]["run"] == str(run)
    finally:
        server.shutdown()
        server.server_close()
