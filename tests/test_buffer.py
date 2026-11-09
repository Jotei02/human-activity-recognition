"""
Equivalence tests for SlidingWindow.append_many vs repeated append().
Run:  python -m pytest tests -q      (or)      python tests/test_buffer.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.buffer import SlidingWindow  # noqa: E402


def _stream(n, t_end, rng, jitter=0.0):
    ts = t_end - (n - 1 - np.arange(n)) * 0.02 + rng.normal(0, jitter, n)
    return list(map(float, ts)), rng.normal(size=(n, 6)).astype(np.float32).tolist()


def _same(a: SlidingWindow, b: SlidingWindow):
    assert a.debug_timestamps() == b.debug_timestamps()
    wa, wb = a.get_window(), b.get_window()
    assert (wa is None) == (wb is None)
    if wa is not None:
        np.testing.assert_array_equal(wa, wb)


def test_in_order_batches_match_single_appends():
    rng = np.random.default_rng(0)
    a, b = SlidingWindow(6), SlidingWindow(6)
    t_end = time.time()
    for k in range(8):                                   # 8 batches of 50 = 8 s of data
        ts, xs = _stream(50, t_end - (7 - k) * 1.0, rng)
        for t, x in zip(ts, xs):
            a.append(t, x)
        b.append_many(ts, xs)
    assert b.is_full() and a.is_full()
    _same(a, b)


def test_out_of_order_and_duplicate_timestamps():
    rng = np.random.default_rng(1)
    a, b = SlidingWindow(6), SlidingWindow(6)
    ts, xs = _stream(300, time.time(), rng, jitter=0.004)   # jitter -> some out of order
    ts[10] = ts[11]                                         # duplicate timestamp
    order = rng.permutation(300)[:300]                      # fully shuffled arrival
    ts = [ts[i] for i in order]; xs = [xs[i] for i in order]
    for t, x in zip(ts, xs):
        a.append(t, x)
    b.append_many(ts, xs)
    assert a.debug_timestamps() == sorted(a.debug_timestamps())
    # same timestamps, same sort order; for ties append() inserts after (bisect_right)
    # and append_many's stable sort keeps arrival order: same result
    _same(a, b)


def test_old_samples_ignored_and_trimmed():
    rng = np.random.default_rng(2)
    a, b = SlidingWindow(6), SlidingWindow(6)
    ts, xs = _stream(1000, time.time() - 5, rng)            # 20 s span: first part too old
    for t, x in zip(ts, xs):
        a.append(t, x)
    b.append_many(ts, xs)
    assert len(b) <= 10 * 50 + 5                            # bounded by max_buffer_seconds
    _same(a, b)


def test_gap_rejects_window_in_both_paths():
    rng = np.random.default_rng(3)
    a, b = SlidingWindow(6), SlidingWindow(6)
    t_end = time.time()
    ts, xs = _stream(200, t_end, rng)
    keep = [i for i in range(200) if not (90 <= i < 110)]   # 0.4 s hole inside the window
    ts = [ts[i] for i in keep]; xs = [xs[i] for i in keep]
    for t, x in zip(ts, xs):
        a.append(t, x)
    b.append_many(ts, xs)
    assert a.is_full() and a.get_window() is None and b.get_window() is None


def test_wrong_channel_count_raises():
    b = SlidingWindow(6)
    try:
        b.append_many([time.time()], [[1, 2, 3]])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
