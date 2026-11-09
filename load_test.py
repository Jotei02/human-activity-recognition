"""
load_test.py
------------
Multi-user load generator for the HAR inference service.

Simulates N concurrent users, each streaming 6-channel IMU samples to /ingest at
50 Hz for a fixed duration, then reads /metrics from the server and writes a
machine-readable summary.

Start the server first, with a metrics window big enough for the whole run:

    METRICS_WINDOW=200000 uvicorn inference.api.main:app --port 8000

Then:

    python load_test.py --users 10 --duration 60 --stream data/processed/train_stream.npy
    python load_test.py --users 1 5 10 25 --duration 60     # sweep; restart server between runs

Notes on reading the results
  * "achieved_rate_hz" is what this client actually managed to send per user. If it
    is noticeably below 50, the load generator (not the server) is the bottleneck
    and the latency numbers understate what the server could face.
  * e2e latency = enqueue in /ingest -> prediction finished (queue wait + windowing +
    features + ONNX). "inference" latency is only the predictor call.
  * Server counters are cumulative since server start, so restart the server
    between runs for clean numbers (the script warns if it detects leftovers).

Requires: pip install aiohttp numpy
"""

import argparse
import asyncio
import json
import time

import aiohttp
import numpy as np

RATE_HZ = 50

# Sample timestamps = wall-clock time at start + monotonic elapsed. A step of the system
# clock mid-run (NTP, WSL time sync) would otherwise show up as a fake gap in the data.
_ANCHOR = time.time() - time.perf_counter()


def wall_now():
    return _ANCHOR + time.perf_counter()

PERIOD = 1.0 / RATE_HZ


def load_stream(path):
    if path:
        stream = np.load(path)
        print(f"Loaded stream {path}: {stream.shape}")
        return stream
    # Synthetic fallback: fine for latency/throughput, meaningless for accuracy.
    print("No --stream given; using a synthetic random-walk signal (latency only).")
    rng = np.random.default_rng(0)
    return np.cumsum(rng.normal(0, 0.05, size=(5000, 6)), axis=0).astype(np.float32)


async def run_user(session, url, user_id, stream, duration, offset, stats, batch=1):
    """One simulated phone: fixed-rate sends for `duration` seconds."""
    # stagger starts so users don't all fire on the same tick
    await asyncio.sleep(offset)
    start = time.perf_counter()
    next_tick = start
    i = user_id * 137  # different start position in the stream per user
    sent = ok = requests = 0
    tick = PERIOD * batch          # one request every batch/50 s carries `batch` samples

    while time.perf_counter() - start < duration:
        if batch == 1:
            s = stream[i % len(stream)]
            i += 1
            payload = {
                "user_id": user_id,
                "ax": float(s[0]), "ay": float(s[1]), "az": float(s[2]),
                "gx": float(s[3]), "gy": float(s[4]), "gz": float(s[5]),
                "ts": wall_now(),  # seconds, matches SlidingWindow
            }
        else:
            # A phone that buffered `batch` samples: timestamps are the real 50 Hz
            # sample times, ending now.
            now = wall_now()
            rows = [stream[(i + k) % len(stream)] for k in range(batch)]
            i += batch
            payload = {
                "user_id": user_id,
                "ts": [now - (batch - 1 - k) * PERIOD for k in range(batch)],
                "samples": [[float(v) for v in r] for r in rows],
            }
        t0 = time.perf_counter()
        try:
            async with session.post(url, json=payload) as resp:
                await resp.read()
                ok += resp.status == 200
                stats["status"][resp.status] = stats["status"].get(resp.status, 0) + 1
                stats["http_ms"].append((time.perf_counter() - t0) * 1000)
        except aiohttp.ClientError:
            stats["errors"] += 1
        sent += batch          # samples delivered (or attempted)
        requests += 1

        next_tick += tick
        delay = next_tick - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        # if delay < 0 we're behind schedule: send immediately (no catch-up burst
        # beyond one request per loop), and achieved_rate_hz will reveal it

    elapsed = time.perf_counter() - start
    stats["per_user"].append({"user_id": user_id, "sent": sent, "ok": ok,
                              "requests": requests,
                              "achieved_rate_hz": round(sent / elapsed, 2),
                              "achieved_req_hz": round(requests / elapsed, 3)})


async def run(args, n_users):
    base = args.url.rstrip("/")
    stream = load_stream(args.stream)
    stats = {"http_ms": [], "errors": 0, "per_user": [], "status": {}}

    conn = aiohttp.TCPConnector(limit=0)  # no client-side connection cap
    async with aiohttp.ClientSession(connector=conn) as session:
        async with session.get(f"{base}/metrics") as r:
            before = await r.json()
        if before["totals"]["queued_samples"] > 0:
            print("WARNING: server already has traffic recorded; restart it for clean numbers.")

        B = args.batch_size
        endpoint = "/ingest" if B == 1 else "/ingest_batch"
        print(f"\n== {n_users} users x {RATE_HZ} Hz for {args.duration}s, batch={B} "
              f"(target {n_users * RATE_HZ} samples/s, {n_users * RATE_HZ / B:.0f} req/s) ==")
        t0 = time.time()
        await asyncio.gather(*[
            run_user(session, f"{base}{endpoint}", uid, stream, args.duration,
                     offset=PERIOD * B * uid / n_users, stats=stats, batch=B)
            for uid in range(n_users)
        ])

        await asyncio.sleep(1.0)  # let the queue drain
        async with session.get(f"{base}/metrics") as r:
            server = await r.json()

    http = np.array(stats["http_ms"]) if stats["http_ms"] else np.array([0.0])
    total_sent = sum(u["sent"] for u in stats["per_user"])          # samples
    total_requests = sum(u["requests"] for u in stats["per_user"])
    result = {
        "users": n_users,
        "batch_size": B,
        "infer_mode_note": "set via server env INFER_MODE (per_batch default, per_sample control)",
        "duration_s": args.duration,
        "target_rate_hz_per_user": RATE_HZ,
        "total_sent": total_sent,
        "total_requests": total_requests,
        "client_errors": stats["errors"],
        "http_status_counts": {str(k): v for k, v in stats["status"].items()},
        "mean_achieved_rate_hz_per_user": round(
            float(np.mean([u["achieved_rate_hz"] for u in stats["per_user"]])), 2),
        # Sum of per-user rates, each over that user's own elapsed time. Dividing by the
        # global wall clock would include the stagger offset and the last batch's
        # trailing interval, understating throughput by up to one batch period.
        "aggregate_req_per_s": round(sum(u["achieved_req_hz"] for u in stats["per_user"]), 1),
        "aggregate_samples_per_s": round(sum(u["achieved_rate_hz"] for u in stats["per_user"]), 1),
        "windows_per_s": round(server["totals"]["inferences"] / args.duration, 1),
        "client_http_ms": {
            "p50": round(float(np.percentile(http, 50)), 3),
            "p99": round(float(np.percentile(http, 99)), 3),
        },
        "server": server,
    }

    bad = {k: v for k, v in stats["status"].items() if k != 200}
    if bad or stats["errors"]:
        print(f"  !! NON-200 RESPONSES / ERRORS: {bad}, client exceptions={stats['errors']}")
    q = server["totals"]["queued_samples"]
    if q != total_sent:
        print(f"  !! sent {total_sent} samples but server queued {q} (lost/rejected: {total_sent - q})")
    inf, e2e = server["latency_ms"], server["e2e_latency_ms"]
    print(f"  achieved rate/user : {result['mean_achieved_rate_hz_per_user']} Hz "
          f"({result['aggregate_samples_per_s']} samples/s, {result['aggregate_req_per_s']} req/s, "
          f"{result['windows_per_s']} windows/s, {stats['errors']} client errors)")
    print(f"  inference latency  : p50={inf['p50']}  p95={inf['p95']}  p99={inf['p99']} ms "
          f"(n={inf['count']})")
    print(f"  end-to-end latency : p50={e2e['p50']}  p95={e2e['p95']}  p99={e2e['p99']} ms")
    print(f"  queue depth        : mean_60s={server['queue_depth']['mean_60s']}  "
          f"max_60s={server['queue_depth']['max_60s']}")
    print(f"  drop rate          : {server['totals']['drop_rate']}")
    print(f"  worker busy (sum)  : {server['worker_utilization'].get('busy_sum', 'n/a')} "
          f"of one event loop  {server['worker_utilization']}")
    if inf["count"] == args.metrics_window_hint:
        print("  NOTE: latency deque is full; raise METRICS_WINDOW so it covers the whole run.")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--users", type=int, nargs="+", default=[10])
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--batch-size", type=int, default=1,
                   help="samples per request. 1 uses /ingest (one POST per sample); "
                        ">1 uses /ingest_batch (e.g. 50 = one request per second per user)")
    p.add_argument("--stream", default=None, help=".npy of shape (T, 6)")
    p.add_argument("--out", default="load_test_results.json")
    p.add_argument("--metrics-window-hint", type=int, default=-1,
                   help="the server's METRICS_WINDOW, to warn if it was too small")
    args = p.parse_args()

    results = []
    for n in args.users:
        results.append(asyncio.run(run(args, n)))
        if len(args.users) > 1 and n != args.users[-1]:
            input("\nRestart the server for a clean run, then press Enter...")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
