"""
simulate_stream.py
------------------
Single-user simulator: replays a recorded 6-channel stream to /ingest at 50 Hz.
For multi-user load testing use load_test.py instead.

Usage
    python simulate_stream.py --stream data/processed/train_stream.npy
"""

import argparse
import time

import numpy as np
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--stream", default="data/processed/train_stream.npy")
parser.add_argument("--url", default="http://localhost:8000/ingest")
parser.add_argument("--user-id", type=int, default=1)
args = parser.parse_args()

stream = np.load(args.stream)
period = 1 / 50  # 50 Hz
next_tick = time.perf_counter()

for sample in stream:
    payload = {
        "user_id": args.user_id,
        "ax": float(sample[0]),
        "ay": float(sample[1]),
        "az": float(sample[2]),
        "gx": float(sample[3]),
        "gy": float(sample[4]),
        "gz": float(sample[5]),
        # SlidingWindow works in SECONDS (window_seconds=2.56, cutoff uses
        # time.time()). Sending milliseconds here collapses the 2.56 s window
        # to 2.56 ms and disables buffer trimming.
        "ts": time.time(),
    }

    resp = requests.post(args.url, json=payload)
    print(resp.status_code, resp.text)

    # fixed-rate pacing: sleep until the next tick instead of a flat 20 ms,
    # so request latency doesn't drag the rate below 50 Hz
    next_tick += period
    time.sleep(max(0.0, next_tick - time.perf_counter()))
