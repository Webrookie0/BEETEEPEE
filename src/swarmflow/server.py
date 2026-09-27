import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import numpy as np

from .baseline import PredictiveController
from .config import GEOMETRIES, WORLDS
from .env import SwarmEnv
from .scenarios import SCENARIOS


def make_server(config, port=8765, checkpoint=None):
    policy = None
    if checkpoint:
        import torch
        from .policy import LearnedController
        torch.set_num_threads(1)
        policy = LearnedController(checkpoint)
        config = policy.config
    state = {"env": SwarmEnv(config), "controller": policy or PredictiveController(config),
             "policy": "learned checkpoint" if policy else "predictive baseline", "seed": 7}
    state["env"].reset(seed=7)
    lock = threading.Lock()
    assets = Path(__file__).parent / "web"

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
            return {**state["env"].snapshot(), "policy": state["policy"], "seed": state["seed"],
                    "config": state["env"].config.to_dict()}

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/api/state":
                with lock:
                    self.respond(self.snapshot())
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
                        capacity = 16 if continuous or world in ("canyon", "pillars", "hills") else env.config.obstacles
                        new_config = replace(env.config, drones=count, geometry=geometry, world=world,
                                             spacing=spacing, duration=float(duration),
                                             obstacles=max(env.config.obstacles, capacity))
                        new_env = SwarmEnv(new_config, scenario)
                        new_env.reset(seed=seed)
                        new_env.continuous = continuous
                        state.update(env=new_env, seed=seed, controller=policy or PredictiveController(new_config))
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


def serve(config, port=8765, checkpoint=None):
    server = make_server(config, port, checkpoint)
    print(f"SwarmFlow: http://127.0.0.1:{server.server_port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
