#!/bin/bash
set -euo pipefail

SWARMFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SWARMFLOW_DIR"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

if ! .venv/bin/python -c 'import swarmflow' >/dev/null 2>&1; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "SwarmFlow needs its Python environment first."
        echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/"
        echo "Then double-click this launcher again."
        if [ -t 0 ]; then read -r -p "Press Enter to close."; fi
        exit 1
    fi
    echo "Setting up SwarmFlow. This first setup needs an internet connection."
    uv sync --locked --python 3.12 --cache-dir "$SWARMFLOW_DIR/.uv-cache"
fi

exec .venv/bin/python - "$SWARMFLOW_DIR" "$@" <<'PY'
import argparse
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

# Talk to 127.0.0.1 directly even when HTTP_PROXY is set in the environment.
urlopen = build_opener(ProxyHandler({})).open

project = Path(sys.argv[1])
parser = argparse.ArgumentParser(description="Start the SwarmFlow viewer (trained model if one exists) and open it.")
parser.add_argument("--no-browser", action="store_true", help="Start or reuse the server without opening a browser")
parser.add_argument("--port", type=int, default=8765, help="Preferred local port (default: 8765)")
args = parser.parse_args(sys.argv[2:])
if not 1024 <= args.port <= 65525:
    parser.error("Choose a port between 1024 and 65525")

def swarmflow_state(port):
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/state", timeout=1) as response:
            data = json.load(response)
        if (isinstance(data, dict) and isinstance(data.get("config"), dict)
                and "positions" in data and "metrics" in data and "policy" in data):
            return data
    except (URLError, OSError, ValueError):
        pass
    return None

def describe(state):
    if state.get("learned_available"):
        return (f"Controller: {state['policy']}. The trained model is loaded; switch Controller in the page "
                "to compare it with the predictive baseline.")
    return "Controller: predictive baseline. No trained model under models/ yet (see guide.md to train one)."

def open_browser(url):
    if not args.no_browser:
        subprocess.run(["open", url], check=True)

port = None
for candidate in range(args.port, args.port + 10):
    state = swarmflow_state(candidate)
    if state is not None:
        url = f"http://127.0.0.1:{candidate}/"
        print(f"SwarmFlow is already running: {url}", flush=True)
        print(describe(state), flush=True)
        open_browser(url)
        sys.exit(0)
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", candidate))
        port = candidate
        break
    except OSError:
        continue
if port is None:
    sys.exit("No local port is available. Try launching with --port 8800.")

url = f"http://127.0.0.1:{port}/"
log_path = project / "runs" / "local-demo.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
process = None

def interrupted(signum, frame):
    raise KeyboardInterrupt

signal.signal(signal.SIGTERM, interrupted)
try:
    with log_path.open("a", buffering=1) as log:
        process = subprocess.Popen([sys.executable, "-m", "swarmflow", "view", "--port", str(port)],
                                   cwd=project, stdout=log, stderr=subprocess.STDOUT)
        state = None
        for _ in range(150):
            if process.poll() is not None:
                sys.exit(f"The simulator could not start. See {log_path}")
            state = swarmflow_state(port)
            if state is not None:
                break
            time.sleep(0.1)
        else:
            sys.exit(f"The simulator did not become ready. See {log_path}")
        print(f"SwarmFlow is ready: {url}", flush=True)
        print(describe(state), flush=True)
        print("Press Run to fly; the model panel below the flight view shows whether it learned correctly.", flush=True)
        print("Keep this Terminal window open. Press Control+C here to stop the server.", flush=True)
        print(f"Log: {log_path}", flush=True)
        open_browser(url)
        process.wait()
except KeyboardInterrupt:
    print("\nStopping this demo server.", flush=True)
finally:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
PY
