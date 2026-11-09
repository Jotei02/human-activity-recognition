"""
fill_readme.py
--------------
Replaces the blocks between <!-- X:START --> and <!-- X:END --> markers in README.md
with tables generated from your real result files, so the README never contains
hand-typed numbers.

    python scripts/fill_readme.py

Reads (all optional; a missing file leaves its block untouched):
    benchmark_results.json    from benchmark.py
    evaluation_results.json   from evaluate.py
    load_test_results.json    from load_test.py
"""

import json
import os
import re

README = "README.md"


def load(path):
    if not os.path.exists(path):
        print(f"  skip: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def bench_table(b):
    names = {"pytorch": "MLP, PyTorch", "onnx_float32": "MLP, ONNX fp32",
             "onnx_int8": "MLP, ONNX int8 (dynamic PTQ)",
             "xgb_native": "XGBoost, native", "xgb_onnx": "XGBoost, ONNX"}
    rows = ["| Runtime | Size (KB) | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (/s) | Macro-F1 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for k, label in names.items():
        if k in b:
            r = b[k]
            rows.append(f"| {label} | {r['size_kb']} | {r['p50_ms']} | {r['p95_ms']} | "
                        f"{r['p99_ms']} | {r['throughput_per_sec']} | {r['macro_f1']} |")
    return "\n".join(rows)


def eval_table(e):
    labels = list(e["mlp"]["per_class"].keys())
    rows = ["| Class | MLP F1 | XGBoost F1 | Both models wrong |",
            "|---|---:|---:|---:|"]
    for lab in labels:
        d = e["class_difficulty"][lab]
        rows.append(f"| {lab} | {e['mlp']['per_class'][lab]['f1']} | "
                    f"{e['xgboost']['per_class'][lab]['f1']} | "
                    f"{d['both_wrong']}/{d['support']} ({d['rate'] * 100:.1f}%) |")
    rows.append(f"| **Overall** | acc {e['mlp']['accuracy']}, macro-F1 {e['mlp']['macro_f1']} | "
                f"acc {e['xgboost']['accuracy']}, macro-F1 {e['xgboost']['macro_f1']} | "
                f"{e['disagreement']['both_wrong']['pct']}% of test windows |")
    return "\n".join(rows)


def worker_busy_sum(server):
    """Sum of per-worker busy fractions = share of the single event loop spent in
    worker code. (fleet_avg divides by the worker count and hides this, since the
    workers are coroutines on one loop and share one core.)"""
    w = server["worker_utilization"]
    if "busy_sum" in w:
        return w["busy_sum"]
    return round(sum(v for k, v in w.items() if k.startswith("worker_")), 3)


def mode_label(r):
    """Single-sample POSTs, or batched; and when batched, how often inference runs."""
    b = r.get("batch_size", 1)
    if b == 1:
        return "1 (per-sample)"
    infer = r["server"].get("config", {}).get("infer_mode", "per_batch")
    return f"{b} ({'infer per sample' if infer == 'per_sample' else 'infer per batch'})"


def load_table(runs):
    rows = ["| Mode (samples/req) | Users | Rate/user | Samples/s | Req/s | Windows/s "
            "| Inference p99 (ms) | End-to-end p50 / p99 (ms) | Max queue | Worker busy (sum) | Drop rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        s = r["server"]
        samples_s = r.get("aggregate_samples_per_s", r["aggregate_req_per_s"])  # old files: 1 sample/req
        windows_s = r.get("windows_per_s",
                          round(s["totals"]["inferences"] / r["duration_s"], 1))
        rows.append(
            f"| {mode_label(r)} | {r['users']} | {r['mean_achieved_rate_hz_per_user']} Hz "
            f"| {samples_s} | {r['aggregate_req_per_s']} | {windows_s} "
            f"| {s['latency_ms']['p99']} | {s['e2e_latency_ms']['p50']} / {s['e2e_latency_ms']['p99']} "
            f"| {s['queue_depth']['max_60s']} | {worker_busy_sum(s)} "
            f"| {s['totals']['drop_rate']} |")
    return "\n".join(rows)


def replace_block(text, name, content):
    pat = re.compile(rf"(<!-- {name}:START -->)(.*?)(<!-- {name}:END -->)", re.S)
    if not pat.search(text):
        print(f"  warning: markers for {name} not found in README")
        return text
    return pat.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(3)}", text)


def main():
    with open(README) as f:
        text = f.read()

    if (b := load("benchmark_results.json")):
        text = replace_block(text, "BENCH", bench_table(b))
    if (e := load("evaluation_results.json")):
        text = replace_block(text, "EVAL", eval_table(e))
    if (l := load("load_test_results.json")):
        text = replace_block(text, "LOAD", load_table(l))

    with open(README, "w") as f:
        f.write(text)
    print("README.md updated.")


if __name__ == "__main__":
    main()
