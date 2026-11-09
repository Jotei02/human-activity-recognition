"""
sweep.py
--------
Runs a list of load-test configurations, each against a FRESH server process so the
cumulative /metrics counters never mix between runs, and writes one combined
load_test_results.json for scripts/fill_readme.py.

Run from the repo root:

    python scripts/sweep.py 10:1 20:1 40:1 \
                            40:50:per_sample 40:50 100:50 200:50 400:50 800:50 \
                            --stream data/processed/train_stream.npy --duration 60

Each config is  USERS:BATCH[:INFER_MODE]
    USERS       simulated phones, each producing 50 samples/s
    BATCH       samples per request. 1 -> POST /ingest, >1 -> POST /ingest_batch
    INFER_MODE  per_batch (default) or per_sample. Only matters when BATCH > 1.
                per_sample runs the same number of model calls as /ingest and is the
                control that separates "fewer HTTP requests" from "fewer predictions".
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request


def wait_ready(port, tries=80):
    for _ in range(tries):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=1)
            return True
        except Exception:
            time.sleep(0.25)
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+", help="USERS:BATCH[:INFER_MODE]")
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--stream", default=None)
    p.add_argument("--out", default="load_test_results.json")
    p.add_argument("--base-port", type=int, default=8100)
    args = p.parse_args()

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, cfg in enumerate(args.configs):
            parts = cfg.split(":")
            users, batch = parts[0], parts[1] if len(parts) > 1 else "1"
            mode = parts[2] if len(parts) > 2 else "per_batch"
            port = args.base_port + k

            # window sized to hold every inference of the run, so percentiles are not
            # computed over just the tail
            window = int(int(users) * 50 * args.duration * 1.2) + 1000
            env = dict(os.environ, METRICS_WINDOW=str(window), INFER_MODE=mode,
                       PYTHONPATH=os.getcwd())
            srv = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "inference.api.main:app",
                 "--port", str(port), "--log-level", "warning"],
                env=env, stderr=subprocess.STDOUT,
                stdout=open(f"server_debug_{k}.log", "w") if os.environ.get("DEBUG_GAPS") else subprocess.DEVNULL)
            try:
                if not wait_ready(port):
                    raise RuntimeError(f"server for {cfg} did not start")
                out = os.path.join(tmp, f"run{k}.json")
                cmd = [sys.executable, "load_test.py", "--url", f"http://localhost:{port}",
                       "--users", users, "--batch-size", batch,
                       "--duration", str(args.duration), "--out", out]
                if args.stream:
                    cmd += ["--stream", args.stream]
                print(f"\n##### {cfg} #####", flush=True)
                subprocess.run(cmd, env=env, check=True)
                results += json.load(open(out))
            finally:
                srv.terminate()
                srv.wait(timeout=10)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} runs -> {args.out}")


if __name__ == "__main__":
    main()
