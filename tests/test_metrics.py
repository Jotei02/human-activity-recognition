"""Regression tests for MetricsCollector clock handling and drop diagnostics."""
import os
import sys
import time
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.buffer import SlidingWindow  # noqa: E402
from utils.metrics import MetricsCollector  # noqa: E402


def test_busy_time_survives_wall_clock_step_back():
    m = MetricsCollector()
    m.worker_registered(0)
    m.worker_start(0)
    real = time.time()
    with mock.patch("time.time", return_value=real - 5.0):   # system clock steps back 5 s
        m.worker_done(0)
    util = m.get_worker_utilization()
    assert util["worker_0"] >= 0 and util["busy_sum"] >= 0, util


def test_gap_drop_reports_gap_size():
    b = SlidingWindow(6)
    now = time.time()
    ts = [now - 4 + i * 0.02 for i in range(100)] + [now - 1.5 + i * 0.02 for i in range(75)]
    for t in ts:
        b.append(t, [0.0] * 6)
    assert b.is_full() and b.get_window() is None
    assert 0.2 < b.last_gap < 0.5 + 0.05, b.last_gap          # ~0.5 s hole
    m = MetricsCollector(); m.record_dropped(b.last_gap)
    assert m.get_summary(0)["totals"]["max_dropped_gap_s"] > 0.2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok ", name)
