import time
import threading
from collections import deque
from typing import Dict, Optional
import numpy as np


class MetricsCollector:
    """
    Thread-safe metrics collector for inference latency, queue depth,
    and worker utilization.

    All latency values are in milliseconds.
    Uses a rolling window so stats reflect recent behaviour, not all-time averages.
    """

    def __init__(self, window_size: int = 500):
        """
        Args:
            window_size: number of recent inference records to keep for percentile calc.
                         500 at 50Hz * 4 workers ≈ last ~2-3 seconds of data.
        """
        self._lock = threading.Lock()

        # Rolling deque of (timestamp, latency_ms) tuples
        self._latencies: deque = deque(maxlen=window_size)

        # End-to-end latency: /ingest enqueue -> prediction finished (includes
        # queue wait, windowing/interpolation, feature extraction, inference).
        self._e2e_latencies: deque = deque(maxlen=window_size)

        # Queue depth snapshots: list of (timestamp, depth) for last 60s
        self._queue_depth_history: deque = deque(maxlen=3000)  # 1 snapshot/20ms * 60s

        # Per-worker busy tracking: worker_id -> busy_since timestamp or None
        self._worker_busy: Dict[int, Optional[float]] = {}
        self._worker_total_busy: Dict[int, float] = {}
        self._worker_total_idle: Dict[int, float] = {}

        # When did the worker last transition state (busy->idle or idle->busy)
        # Used to calculate how long it's been in current state
        self._worker_last_transition: Dict[int, float] = {}

        # Counters
        self._total_inferences: int = 0
        self._max_dropped_gap: float = 0.0
        self._total_dropped: int = 0  # windows rejected (gap / not full)
        self._total_queued: int = 0

        self._start_time = time.monotonic()

    # ------------------------------------------------------------------
    # Latency tracking
    # ------------------------------------------------------------------

    def record_latency(self, latency_ms: float):
        """Call after each predictor.predict() with elapsed ms."""
        with self._lock:
            self._latencies.append((time.monotonic(), latency_ms))
            self._total_inferences += 1

    def record_e2e_latency(self, latency_ms: float):
        """Call after a prediction with ms elapsed since the sample was enqueued."""
        with self._lock:
            self._e2e_latencies.append((time.monotonic(), latency_ms))

    def record_dropped(self, gap_s: float = 0.0):
        """Call when get_window() returns None. gap_s is the timestamp gap that caused
        it; the max is reported so a drop can be traced to a stall vs a clock jump."""
        with self._lock:
            self._total_dropped += 1
            self._max_dropped_gap = max(self._max_dropped_gap, gap_s)

    def record_queued(self, n_samples: int = 1):
        """Call inside /ingest or /ingest_batch with the number of samples enqueued."""
        with self._lock:
            self._total_queued += n_samples

    # ------------------------------------------------------------------
    # Queue depth snapshots
    # ------------------------------------------------------------------

    def record_queue_depth(self, depth: int):
        """
        Call on every worker iteration with current queue size.
        Automatically called by worker_start() and worker_done() so you
        don't need to call it manually if you use those.
        """
        with self._lock:
            self._queue_depth_history.append((time.monotonic(), depth))

    # ------------------------------------------------------------------
    # Worker utilization
    # ------------------------------------------------------------------

    def worker_registered(self, worker_id: int):
        """
        Call once when a worker starts up, before its first queue.get().
        Initialises all tracking state for this worker.
        """
        now = time.monotonic()
        with self._lock:
            self._worker_busy[worker_id] = None  # starts idle
            self._worker_total_busy[worker_id] = 0.0
            self._worker_total_idle[worker_id] = 0.0
            self._worker_last_transition[worker_id] = now

    def worker_start(self, worker_id: int):
        """
        Call when a worker picks up a task (after queue.get()).
        Closes out the idle period and opens a busy period.
        """
        now = time.monotonic()
        with self._lock:
            last = self._worker_last_transition.get(worker_id, now)

            # accumulate idle time since last transition
            self._worker_total_idle[worker_id] = (
                self._worker_total_idle.get(worker_id, 0.0) + (now - last)
            )

            self._worker_busy[worker_id] = now
            self._worker_last_transition[worker_id] = now

    def worker_done(self, worker_id: int):
        """
        Call when a worker finishes processing and calls task_done().
        Closes out the busy period and opens an idle period.
        """
        now = time.monotonic()
        with self._lock:
            last = self._worker_last_transition.get(worker_id, now)

            # accumulate busy time since last transition
            self._worker_total_busy[worker_id] = (
                self._worker_total_busy.get(worker_id, 0.0) + (now - last)
            )

            self._worker_busy[worker_id] = None
            self._worker_last_transition[worker_id] = now

    # ------------------------------------------------------------------
    # Stats computation
    # ------------------------------------------------------------------

    def get_latency_stats(self) -> Dict:
        """
        Returns p50/p95/p99/mean/max over the rolling window.
        Returns zeros if no data yet.
        """
        with self._lock:
            if not self._latencies:
                return {"p50": 0, "p95": 0, "p99": 0, "mean": 0, "max": 0, "count": 0}
            values = [lat for _, lat in self._latencies]

        arr = np.array(values)
        return {
            "p50":  round(float(np.percentile(arr, 50)), 3),
            "p95":  round(float(np.percentile(arr, 95)), 3),
            "p99":  round(float(np.percentile(arr, 99)), 3),
            "mean": round(float(np.mean(arr)), 3),
            "max":  round(float(np.max(arr)), 3),
            "count": len(arr),
        }

    def get_e2e_latency_stats(self) -> Dict:
        """Same percentiles as get_latency_stats, for enqueue -> prediction time."""
        with self._lock:
            if not self._e2e_latencies:
                return {"p50": 0, "p95": 0, "p99": 0, "mean": 0, "max": 0, "count": 0}
            values = [lat for _, lat in self._e2e_latencies]

        arr = np.array(values)
        return {
            "p50":  round(float(np.percentile(arr, 50)), 3),
            "p95":  round(float(np.percentile(arr, 95)), 3),
            "p99":  round(float(np.percentile(arr, 99)), 3),
            "mean": round(float(np.mean(arr)), 3),
            "max":  round(float(np.max(arr)), 3),
            "count": len(arr),
        }

    def get_queue_stats(self, queue_size: int) -> Dict:
        """
        Returns current, mean, and max queue depth over the last 60s.
        Pass in the live queue size so current is always accurate.
        """
        now = time.monotonic()

        # record the live depth before computing stats
        self.record_queue_depth(queue_size)

        with self._lock:
            if not self._queue_depth_history:
                return {"current": queue_size, "mean_60s": 0.0, "max_60s": 0}

            recent = [d for ts, d in self._queue_depth_history if now - ts <= 60]

        if not recent:
            return {"current": queue_size, "mean_60s": 0.0, "max_60s": 0}

        return {
            "current":  queue_size,
            "mean_60s": round(float(np.mean(recent)), 2),
            "max_60s":  int(np.max(recent)),
        }

    def get_worker_utilization(self) -> Dict:
        """
        Returns utilization ratio (0.0–1.0) per worker, plus a fleet average.
        Utilization = busy_time / (busy_time + idle_time).
        Workers currently busy have their in-progress time included.
        """
        now = time.monotonic()
        with self._lock:
            result = {}
            for wid in self._worker_total_busy:
                busy = self._worker_total_busy.get(wid, 0.0)
                idle = self._worker_total_idle.get(wid, 0.0)

                # if worker is currently busy, add the in-progress slice
                if self._worker_busy.get(wid) is not None:
                    busy += now - self._worker_last_transition.get(wid, now)
                else:
                    idle += now - self._worker_last_transition.get(wid, now)

                total = busy + idle
                util = busy / total if total > 0 else 0.0
                result[f"worker_{wid}"] = round(util, 3)

        if result:
            vals = list(result.values())
            result["fleet_avg"] = round(float(np.mean(vals)), 3)
            # The workers are coroutines on ONE event loop, so the mean over workers
            # hides load. The sum is the share of that single loop spent in worker
            # code (the rest is HTTP parsing, validation, etc.).
            result["busy_sum"] = round(float(np.sum(vals)), 3)

        return result

    def get_summary(self, queue_size: int) -> Dict:
        """
        Single call returning everything — used by /metrics endpoint.
        Pass in sensor_queue.qsize() so queue stats are live.
        """
        uptime = round(time.monotonic() - self._start_time, 1)
        with self._lock:
            total_inf = self._total_inferences
            total_drop = self._total_dropped
            max_gap = self._max_dropped_gap
            total_q = self._total_queued

        drop_rate = (
            round(total_drop / (total_inf + total_drop), 4)
            if (total_inf + total_drop) > 0
            else 0.0
        )

        return {
            "uptime_seconds": uptime,
            "totals": {
                "inferences": total_inf,
                "dropped_windows": total_drop,
                "queued_samples": total_q,
                "drop_rate": drop_rate,
                "max_dropped_gap_s": round(max_gap, 3),
            },
            "latency_ms": self.get_latency_stats(),
            "e2e_latency_ms": self.get_e2e_latency_stats(),
            "queue_depth": self.get_queue_stats(queue_size),
            "worker_utilization": self.get_worker_utilization(),
        }
