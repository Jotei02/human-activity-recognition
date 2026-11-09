# Future work

Prioritised by how much each item improves the project for the effort. Items marked **(known issue)** are open problems from the README.

## 1. Close the open questions (highest value, small effort)

1. **Explain the dropped windows in batched mode (known issue).**
   Run the server with `DEBUG_GAPS=1` in one terminal and `python load_test.py --users 40 --batch-size 50 --duration 60` in another. Read the `[GAP]` lines: the same `batch#` on every user with an arrival interval of about 2 s means a global stall; an arrival interval of about 1 s with a timestamp jump means a client timestamp problem. If it is a stall, stamp batch samples from the schedule (`t0 + n_made * 0.02`) instead of from the clock at send time, as a phone's sensor clock would. Rerun and confirm drops go to 0.
2. **Control experiment for the bottleneck (known issue).**
   Run `python scripts/sweep.py 80:1 80:50:per_sample 100:50:per_sample`. If `per_sample` batched mode holds 50 Hz where single-sample collapses, HTTP overhead was the limit. If it collapses too, per-sample prediction cost is the limit. Then rewrite the batched-ingestion paragraph with the answer.
3. **Repeat each quoted load-test row 3 times** and report the median with the range. The 40-user p99 varied between 2 and 8 ms across runs.
4. **Train/serve parity test.** Replay the UCI test windows through the live service and compare predictions with the offline ones. This checks that timestamp interpolation and feature extraction give the same inputs the model was trained on. It is the most important correctness test missing.

## 2. Serving performance

5. **Profile before optimising.** Use `py-spy` or `cProfile` on a saturated run to find where the worker time goes.
6. **Fix the O(n) work per sample in `SlidingWindow`.** `append` rebuilds the list of timestamps on every insert and trim. Keep a separate sorted key list, or use a preallocated numpy ring buffer with a timestamp array.
7. **Configurable prediction stride.** Predict every N samples (for example 50% window overlap) instead of after every sample. This is where most of the batched-mode gain comes from, so make it explicit and measure accuracy against latency.
8. **Move CPU work off the event loop.** Try `run_in_executor` or a process pool for feature extraction and inference, and compare.
9. **Scale out.** Several uvicorn processes with sticky routing by `user_id`, or per-user buffers in Redis. Re-run the sweep and report scaling efficiency.
10. **Backpressure and limits.** Bounded queue returning 429 or 503 when full, evicting buffers of idle users (the `buffers` dict currently grows with every user ID), and rejecting NaN or out-of-range sensor values.

## 3. Product completeness

11. **Return predictions.** Store the latest label and confidence per user and expose `GET /predict/{user_id}`, or a WebSocket stream. Today predictions are computed but only counted.
12. **Health and versioning.** `/healthz`, plus the model version and git commit in `/metrics`.
13. **Observability.** Export Prometheus metrics, add structured JSON logs, and optionally a small Grafana dashboard. A screenshot under load makes a strong README image.
14. **Small demo.** A page or script that streams a recorded session and shows the predicted activity changing live. A short GIF in the README is worth more than another table.

## 4. Modelling

15. **Subject-wise cross-validation** and variance across folds, not a single train/test split.
16. **Address sitting vs standing.** Try gravity-direction and tilt-angle features from the low-pass accelerometer, and frequency-domain features. Report whether the confusion actually drops, using the EDA feature-separation numbers as the baseline.
17. **Compare against a small 1D-CNN** on the raw 128 x 6 windows. Report accuracy, size and latency alongside the MLP and XGBoost.
18. **Calibrate confidence** (temperature scaling) and add an "uncertain" output below a threshold, since the service already returns a confidence.
19. **Quantization trade-off.** Report int8 accuracy and latency deltas per class, and check whether int8 helps for a model this small.

## 5. Engineering hygiene

20. **CI:** GitHub Actions running lint, `tests/`, and `docker build` on every push.
21. **Reproducibility:** a `Makefile` (`make data`, `make train`, `make bench`, `make loadtest`), pinned dependency versions or a lockfile, and a script that downloads the UCI dataset.
22. **Docker:** non-root user, `HEALTHCHECK`, pinned base image, a compose file that runs the service and the load test together, and publishing the image to GHCR.
23. **Results archive:** commit the result JSON files and plots under `docs/results/` with the date and machine spec, so every number in the README is checkable.
24. **Type hints and `ruff`/`black`** across the modules, and a `LICENSE`.

## Suggested order

1 to 4 first (they settle the open claims), then 11 and 14 (the project becomes demonstrable), then 20 to 23, then performance and modelling items as time allows.
