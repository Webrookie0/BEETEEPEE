import json
import os
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import numpy as np

from .baseline import PredictiveController, SlotController
from .config import GEOMETRIES, WORLDS
from .env import SwarmEnv
from .scenarios import SCENARIOS


CONTROLLER_LABELS = {"learned": "learned checkpoint", "baseline": "predictive baseline",
                     "slot": "slot-only (no avoidance)"}


def _load(path):
    import torch
    from .policy import LearnedController
    torch.set_num_threads(1)
    return LearnedController(path)


def _stamp(path):
    return str(path), os.path.getmtime(path)


def make_server(config, port=8765, checkpoint=None, follow=False, models_dir="models", interval=5.0):
    """With follow=True the viewer keeps switching to the best model under models_dir as training saves it."""
    state = {"learned": None, "checkpoint": None, "stamp": None, "seed": 7}
    if checkpoint:
        state.update(learned=_load(checkpoint), checkpoint=str(checkpoint), stamp=_stamp(checkpoint))
        config = state["learned"].config
    base = config

    def controller_for(mode, cfg):
        if mode == "learned":
            return state["learned"]
        return SlotController(cfg) if mode == "slot" else PredictiveController(cfg)

    initial = "learned" if state["learned"] else "baseline"
    state.update(env=SwarmEnv(config), controller=controller_for(initial, config), mode=initial,
                 policy=CONTROLLER_LABELS[initial])
    state["env"].reset(seed=7)
    lock = threading.Lock()
    assets = Path(__file__).parent / "web"

    def follow_best():
        from .curriculum import default_model
        while True:
            time.sleep(interval)
            try:
                found = default_model(models_dir)
                stamp = _stamp(found) if found else None
                if stamp is None or stamp == state["stamp"]:
                    continue
                fresh = _load(found)
            except (OSError, ValueError, KeyError, RuntimeError):
                continue
            with lock:
                first = state["learned"] is None
                state.update(learned=fresh, checkpoint=str(found), stamp=stamp)
                if state["mode"] == "learned" or (first and state["mode"] == "baseline"):
                    state.update(controller=fresh, mode="learned", policy=CONTROLLER_LABELS["learned"])
            print(f"Viewer now uses {found}", flush=True)

    if follow:
        threading.Thread(target=follow_best, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def respond(self, value, status=200, content_type="application/json"):
            body = json.dumps(value, allow_nan=False).encode() if content_type == "application/json" else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def snapshot(self):
            return {**state["env"].snapshot(), "policy": state["policy"], "mode": state["mode"],
                    "learned_available": state["learned"] is not None, "checkpoint": state["checkpoint"],
                    "seed": state["seed"], "config": state["env"].config.to_dict()}

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/api/state":
                with lock:
                    self.respond(self.snapshot())
                return
            if route == "/api/model":
                from .curriculum import model_report, training_status
                live = training_status(models_dir)
                path = state["checkpoint"]
                if not path:
                    self.respond({"available": False, "training": live})
                    return
                try:
                    self.respond({"available": True, **model_report(path), "training": live})
                except (OSError, ValueError, KeyError) as error:
                    self.respond({"available": True, "checkpoint": path, "error": str(error), "training": live})
                return
            if route == "/api/tests":
                from .suite import catalog
                self.respond({"tests": catalog(), "defaults": {"geometry": base.geometry, "world": base.world,
                                                             "spacing": base.spacing,
                                                             "obstacles": base.obstacles}})
                return
            files = {"/": ("index.html", "text/html; charset=utf-8"),
                     "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                     "/style.css": ("style.css", "text/css; charset=utf-8")}
            if route not in files:
                self.respond({"error": "Not found"}, 404)
                return
            name, mime = files[route]
            self.respond((assets / name).read_bytes(), content_type=mime)

        def do_POST(self):
            try:
                origin = self.headers.get("Origin")
                if origin and origin != f"http://{self.headers.get('Host')}":
                    self.respond({"error": "Cross-origin requests are not accepted"}, 403)
                    return
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                with lock:
                    env = state["env"]
                    if self.path == "/api/reset":
                        scenario = data.get("scenario", env.scenario)
                        count = data.get("drones", env.config.drones)
                        seed = data.get("seed", state["seed"])
                        geometry = data.get("geometry", env.config.geometry)
                        world = data.get("world", env.config.world)
                        spacing = data.get("spacing", env.config.spacing)
                        if scenario == "terrain_follow":
                            world = "hills"
                        continuous = bool(data.get("continuous", False))
                        duration = data.get("duration", 3600.0 if continuous else (180.0 if scenario == "pilot" else 24.0))
                        if scenario not in SCENARIOS:
                            raise ValueError(f"Unknown scenario {scenario!r}")
                        try:
                            seed = int(seed)
                            count = int(count)
                            spacing = float(spacing)
                        except (TypeError, ValueError) as error:
                            raise ValueError("Seed, drones and spacing must be numeric") from error
                        if isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
                            raise ValueError(f"Invalid seed {seed}")
                        if geometry not in GEOMETRIES:
                            raise ValueError(f"geometry must be one of {GEOMETRIES}")
                        if world not in WORLDS:
                            raise ValueError(f"world must be one of {WORLDS}")
                        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not 1 <= float(duration) <= 7200:
                            raise ValueError("duration must be between 1 and 7200 seconds")
                        controller = data.get("controller", state["mode"])
                        if controller not in CONTROLLER_LABELS:
                            raise ValueError("controller must be 'learned', 'baseline' or 'slot'")
                        if controller == "learned" and not state["learned"]:
                            raise ValueError("No trained policy is loaded; train one (make training) or "
                                             "start the viewer with --checkpoint")
                        sensing = {}
                        for key in ("noise", "dropout"):
                            value = data.get(key, getattr(base, key))
                            if not isinstance(value, (int, float)) or isinstance(value, bool):
                                raise ValueError(f"{key} must be numeric")
                            sensing[key] = float(value)
                        requested = data.get("obstacles", base.obstacles)
                        if not isinstance(requested, int) or isinstance(requested, bool):
                            raise ValueError("obstacles must be an integer")
                        capacity = 16 if continuous or world in ("canyon", "pillars", "hills") else requested
                        new_config = replace(env.config, drones=count, geometry=geometry, world=world,
                                             spacing=spacing, duration=float(duration),
                                             obstacles=max(requested, capacity), **sensing)
                        new_env = SwarmEnv(new_config, scenario)
                        new_env.reset(seed=seed)
                        new_env.continuous = continuous
                        state.update(env=new_env, seed=seed, controller=controller_for(controller, new_config),
                                     mode=controller, policy=CONTROLLER_LABELS[controller])
                    elif self.path == "/api/step":
                        count = data.get("steps", 1)
                        if not isinstance(count, int) or not 1 <= count <= 4:
                            raise ValueError("steps must be an integer between 1 and 4")
                        for _ in range(count):
                            if env.agents:
                                env.step(state["controller"](env.last_observations))
                    elif self.path == "/api/inject":
                        env.inject(float(data.get("speed", 1.6)), float(data.get("radius", 0.7)),
                                   float(data.get("angle", np.pi / 2)))
                    elif self.path == "/api/steer":
                        velocity = data.get("velocity")
                        position = data.get("position")
                        env.steer(velocity=velocity, position=position)
                    else:
                        self.respond({"error": "Not found"}, 404)
                        return
                    self.respond(self.snapshot())
            except (ValueError, TypeError, OverflowError) as error:
                self.respond({"error": str(error)}, 400)
            except Exception:
                self.respond({"error": "Simulation failed. Reset the scenario and check the terminal."}, 500)
                raise

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(config, port=8765, checkpoint=None, follow=False):
    server = make_server(config, port, checkpoint, follow)
    print(f"SwarmFlow: http://127.0.0.1:{server.server_port} (Ctrl+C to stop)", flush=True)
    print(f"Controller: {'learned checkpoint ' + str(checkpoint) if checkpoint else 'predictive baseline'}",
          flush=True)
    if follow:
        print("Following models/: the viewer switches to each new best model automatically", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
