from bisect import bisect_right
from typing import List, Tuple, Optional
import numpy as np
import time

class SlidingWindow:
    """
    Time-aware sliding window.

    - Stores samples as (timestamp, sample_array) in a sorted list.
    - append_with_timestamp inserts samples (handles out-of-order).
    - Keeps samples within `max_buffer_seconds` (trimming old ones).
    - Can resample the most-recent `window_seconds` into `n_samples`
      evenly-spaced time points via linear interpolation across each channel.
    """

    def __init__(self,
                 n_channels: int,
                 window_seconds: float = 2.56,
                 n_samples: int = 128,
                 max_buffer_seconds: float = 10.0,
                 max_allowed_gap: float = 0.2):
        """
        Args:
            n_channels: number of sensor channels (6 for accel+gyro)
            window_seconds: target window duration (seconds) e.g. 2.56
            n_samples: number of samples to output per window (e.g. 128)
            max_buffer_seconds: max history to keep (seconds)
            max_allowed_gap: if any gap > this (seconds) inside the target window,
                             consider window invalid (to avoid bad interpolation)
        """
        self.n_channels = n_channels
        self.window_seconds = window_seconds
        self.n_samples = n_samples
        self.max_buffer_seconds = max_buffer_seconds
        self.max_allowed_gap = max_allowed_gap

        # underlying storage: list of (ts, 1d-array(length=n_channels))
        self._data: List[Tuple[float, np.ndarray]] = []
        self.last_gap: float = 0.0  # largest gap (s) that caused the latest rejected window

    # -------------------------
    # Ingestion / maintenance
    # -------------------------
    def append(self, ts: float, sample: List[float]):
        """
        Insert sample at timestamp `ts`. Handles out-of-order inserts.
        If ts is older than the retention window, it will be ignored.
        """
        if len(sample) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {len(sample)}")

        # normalize ts
        ts = float(ts)
        arr = np.asarray(sample, dtype=np.float32)

        # Ignore samples that are far too old (older than max_buffer_seconds before now)
        cutoff = time.time() - self.max_buffer_seconds
        if ts < cutoff:
            # too old to be useful
            return

        # find insertion index to keep list sorted by timestamp
        keys = [t for t, _ in self._data]
        idx = bisect_right(keys, ts)

        # small guard: if same timestamp exists, insert after (bisect_right)
        self._data.insert(idx, (ts, arr))

        # trim old samples beyond retention window
        self._trim()

    def append_many(self, timestamps: List[float], samples: List[List[float]]):
        """
        Insert a batch in one pass: one sort and one trim instead of one
        bisect/insert/trim per sample. Same result as calling append() in order
        (stable sort keeps arrival order for equal timestamps).
        """
        if len(timestamps) != len(samples):
            raise ValueError("timestamps and samples must have equal length")
        cutoff = time.time() - self.max_buffer_seconds
        new = []
        for ts, sample in zip(timestamps, samples):
            if len(sample) != self.n_channels:
                raise ValueError(f"Expected {self.n_channels} channels, got {len(sample)}")
            ts = float(ts)
            if ts >= cutoff:
                new.append((ts, np.asarray(sample, dtype=np.float32)))
        if not new:
            return
        self._data.extend(new)
        self._data.sort(key=lambda item: item[0])   # timsort: ~O(n) on nearly-sorted data
        self._trim()

    def _trim(self):
        """Remove samples older than now - max_buffer_seconds to bound memory."""
        if not self._data:
            return
        cutoff = time.time() - self.max_buffer_seconds
        # find first index with ts >= cutoff
        keys = [t for t, _ in self._data]
        i = bisect_right(keys, cutoff - 1e-9)  # keep anything >= cutoff
        if i > 0:
            # drop first i elements
            self._data = self._data[i:]

    # -------------------------
    # Window readiness / retrieval
    # -------------------------
    def is_full(self) -> bool:
        """Return True if buffer contains any samples within last window_seconds."""
        if not self._data:
            return False
        latest_ts = self._data[-1][0]
        earliest_needed = latest_ts - self.window_seconds
        return self._data[0][0] <= earliest_needed

    def get_window(self) -> Optional[np.ndarray]:
        """
        Return a (n_samples, n_channels) numpy array representing the most recent
        window_seconds, resampled to exactly n_samples via linear interpolation.

        Returns None if:
         - there is not enough data, or
         - there is a gap larger than max_allowed_gap inside the target window
        """
        if not self._data:
            return None

        latest_ts = self._data[-1][0]
        start_ts = latest_ts - self.window_seconds

        # select data within [start_ts, latest_ts]
        keys = [t for t, _ in self._data]
        left = bisect_right(keys, start_ts - 1e-9)  # first index with ts >= start_ts
        window_data = self._data[left:]

        if not window_data:
            return None

        timestamps = np.array([t for t, _ in window_data], dtype=np.float64)
        values = np.stack([v for _, v in window_data], axis=0)  # shape (M, C)

        # quick check: if first timestamp is after start_ts, we'll interpolate from first point
        # check for any big gaps (possible missing data)
        gaps = np.diff(timestamps)
        if gaps.size > 0 and np.max(gaps) > self.max_allowed_gap:
            self.last_gap = float(np.max(gaps))   # for drop diagnostics in /metrics
            return None  # too much gap inside window

        # define target timestamps: n_samples evenly spaced in [start_ts, latest_ts]
        target_ts = np.linspace(start_ts, latest_ts, num=self.n_samples, dtype=np.float64)

        # For each channel, interpolate
        # If there is only one sample, repeat it
        if timestamps.size == 1:
            resampled = np.repeat(values, self.n_samples, axis=0)
            return resampled.astype(np.float32)

        resampled = np.empty((self.n_samples, self.n_channels), dtype=np.float32)
        for ch in range(self.n_channels):
            ch_vals = values[:, ch].astype(np.float64)
            # np.interp requires timestamps to be increasing
            resampled[:, ch] = np.interp(target_ts, timestamps, ch_vals).astype(np.float32)

        return resampled

    # -------------------------
    # Utility helpers
    # -------------------------
    def clear(self):
        self._data = []

    def __len__(self):
        return len(self._data)

    def debug_timestamps(self) -> List[float]:
        return [t for t, _ in self._data]
