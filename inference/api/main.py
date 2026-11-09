import asyncio
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager

from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils.buffer import SlidingWindow
from utils.features import extract_features
from utils.metrics import MetricsCollector
from utils.predictor import ONNXPredictor
from inference.api.queue import sensor_queue

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Shared state
# ------------------------------------------------------------------

buffers = defaultdict(lambda: SlidingWindow(n_channels=6, n_samples=128))
predictor = ONNXPredictor("model/mlp_model.onnx", "model/scaler.pkl")
# Warm-up: the first ONNX Runtime calls allocate buffers / initialise kernels and are
# several times slower than steady state. Do them before serving so that one-off cost
# doesn't show up as a latency outlier in /metrics (it dominated p99 for short runs).
for _ in range(20):
    predictor.predict(np.zeros((1, 13), dtype=np.float32))
# Rolling window for latency percentiles. Default 500 suits interactive use; for a
# load test set METRICS_WINDOW high enough to cover the whole run
# (e.g. users * 50 Hz * seconds).
metrics = MetricsCollector(window_size=int(os.getenv("METRICS_WINDOW", "500")))

NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))

# How often a *batched* request triggers inference:
#   per_batch  - append the whole batch, then run one window/inference (default;
#                a phone sending 1 s of data wants one fresh prediction, not 50)
#   per_sample - append and infer after every sample in the batch, i.e. the same
#                model compute as /ingest. Used as a control in the load test to
#                separate the HTTP-batching gain from the fewer-predictions gain.
INFER_MODE = os.getenv("INFER_MODE", "per_batch")
MAX_BATCH = 500


# ------------------------------------------------------------------
# App lifecycle
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = []
    for worker_id in range(NUM_WORKERS):
        task = asyncio.create_task(worker(worker_id))
        tasks.append(task)
    
    yield
    
    # shutdown
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)  # wait for all to finish cancelling
    
    snapshot = await get_metrics()
    print("Final metrics snapshot:", snapshot)


app = FastAPI(lifespan=lifespan)


# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------

class SensorReading(BaseModel):
    user_id: int
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    ts: float


class SensorBatch(BaseModel):
    """Compact columnar batch: one validation pass over plain float lists instead of
    one pydantic model per sample. samples[i] = [ax, ay, az, gx, gy, gz] at ts[i]."""
    user_id: int
    ts: List[float]
    samples: List[List[float]]


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
# Queue items are (user_id, timestamps, samples, enqueued_at) for both endpoints, so
# the worker has a single code path. enqueued_at lets it report end-to-end latency.

@app.post("/ingest")
async def ingest(data: SensorReading):
    sample = [data.ax, data.ay, data.az, data.gx, data.gy, data.gz]
    await sensor_queue.put((data.user_id, [data.ts], [sample], time.perf_counter()))
    metrics.record_queued(1)
    return {"status": "queued"}


@app.post("/ingest_batch")
async def ingest_batch(batch: SensorBatch):
    n = len(batch.ts)
    if n == 0 or n > MAX_BATCH:
        raise HTTPException(422, f"batch size must be 1..{MAX_BATCH}, got {n}")
    if len(batch.samples) != n or any(len(s) != 6 for s in batch.samples):
        raise HTTPException(422, "samples must be a list of len(ts) rows with 6 channels each")
    await sensor_queue.put((batch.user_id, batch.ts, batch.samples, time.perf_counter()))
    metrics.record_queued(n)
    return {"status": "queued", "samples": n}


@app.get("/metrics")
async def get_metrics():
    """
    Returns:
        uptime, total inference/drop counts, drop rate,
        latency percentiles (p50/p95/p99/mean/max in ms),
        queue depth (current / mean_60s / max_60s),
        per-worker utilization (0.0 = idle, 1.0 = always busy)

    Interpretation guide
    --------------------
    p99 latency < 20ms   → comfortably within 50Hz (20ms/sample) budget
    drop_rate   < 0.02   → less than 2% of windows rejected (gaps/no data)
    queue max_60s > 100  → workers can't keep up; consider adding workers
    worker util  > 0.9   → worker is a bottleneck; scale out
    """
    summary = metrics.get_summary(sensor_queue.qsize())
    summary["config"] = {"infer_mode": INFER_MODE, "workers": NUM_WORKERS}
    return summary

# ------------------------------------------------------------------
# Worker
# ------------------------------------------------------------------

def infer_if_ready(user_id: int, buf: SlidingWindow, enqueued_at: float):
    """Run window -> features -> ONNX if the buffer holds a valid window."""
    if not buf.is_full():
        return
    window = buf.get_window()
    if window is None:
        metrics.record_dropped(buf.last_gap)
        return

    features = extract_features(window)

    t0 = time.perf_counter()
    label, confidence = predictor.predict(features)
    latency_ms = (time.perf_counter() - t0) * 1000

    metrics.record_latency(latency_ms)
    # NOTE: server-side only. For batched requests it excludes the time the phone
    # spent collecting the batch (up to batch_size / 50 s).
    metrics.record_e2e_latency((time.perf_counter() - enqueued_at) * 1000)
    # debug, not info/print: at 50Hz x N users this fires thousands
    # of times a minute and the I/O would distort the load test.
    logger.debug("[USER %d] %s (conf=%.2f, latency=%.2fms)",
                 user_id, label, confidence, latency_ms)


DEBUG_GAPS = os.getenv("DEBUG_GAPS") == "1"
_dbg = {"prev": {}, "n": {}, "printed": 0}


def debug_continuity(user_id, tss):
    """Opt-in (DEBUG_GAPS=1): print where a user's sample stream has a hole, and
    whether the batch ARRIVED late/missing (arrival interval ~2 s) or arrived on time
    with a timestamp jump (client/clock problem)."""
    now = time.perf_counter()
    k = _dbg["n"][user_id] = _dbg["n"].get(user_id, 0) + 1
    prev = _dbg["prev"].get(user_id)
    if prev is not None and _dbg["printed"] < 40:
        jump = min(tss) - prev[0]
        if jump > 0.1:
            _dbg["printed"] += 1
            print(f"[GAP] user={user_id} batch#{k} ts_jump={jump:.3f}s "
                  f"arrival_interval={now - prev[1]:.3f}s n={len(tss)} "
                  f"wallclock_minus_last_ts={time.time() - max(tss):.3f}s", flush=True)
    _dbg["prev"][user_id] = (max(tss), now)


async def worker(worker_id: int):
    metrics.worker_registered(worker_id)  # initialise tracking before the loop
    while True:
        try:
            user_id, tss, samples, enqueued_at = await sensor_queue.get()
            metrics.worker_start(worker_id)
            metrics.record_queue_depth(sensor_queue.qsize())

            buf = buffers[user_id]
            if DEBUG_GAPS:
                debug_continuity(user_id, tss)
            if len(tss) == 1 or INFER_MODE == "per_sample":
                for ts, sample in zip(tss, samples):
                    buf.append(ts, sample)
                    infer_if_ready(user_id, buf, enqueued_at)
            else:
                buf.append_many(tss, samples)
                infer_if_ready(user_id, buf, enqueued_at)

        except Exception:
            logger.exception("Worker %d error", worker_id)

        finally:
            sensor_queue.task_done()  # moved here — always called
            metrics.worker_done(worker_id)
