"""
benchmark.py
------------
Compares five inference runtimes across two model families:

  MLP (PyTorch)
    1. PyTorch native       (.pth  loaded into model.eval())
    2. ONNX float32         (.onnx as exported from torch.onnx.export)
    3. ONNX int8 PTQ        (dynamically quantized from the float32 ONNX)

  XGBoost
    4. XGBoost native       (.json loaded with xgb.XGBClassifier)
    5. XGBoost ONNX         (.onnx exported via onnxmltools)

Metrics reported per runtime
  - Model file size (KB)
  - Latency: mean, p50, p95, p99  (milliseconds, over N_RUNS warmup-excluded calls)
  - Macro F1 + per-class F1 on the held-out test set
  - Throughput: inferences / second

Usage
  python benchmark.py

Outputs
  benchmark_results.json   machine-readable summary
  benchmark_results.txt    human-readable table (paste into README)
"""

import time
import json
import os
import numpy as np
import torch
import joblib
import xgboost as xgb
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from sklearn.metrics import f1_score, classification_report
from model import MLP

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

# MLP paths
ONNX_FLOAT32_PATH  = "model/mlp_model.onnx"
ONNX_PREP_PATH     = "model/mlp_model_prep.onnx"   # ORT-preprocessed intermediate
ONNX_INT8_PATH     = "model/mlp_model_int8.onnx"
PTH_PATH           = "model/mlp.pth"

# XGBoost paths
XGB_JSON_PATH      = "model/xgb_model.json"
XGB_ONNX_PATH      = "model/xgb_model.onnx"

SCALER_PATH        = "model/scaler.pkl"
X_TEST_PATH        = "data/processed/X_test.npy"
Y_TEST_PATH        = "data/processed/y_test.npy"

N_WARMUP = 50    # calls to discard before timing (JIT / session warmup)
N_RUNS   = 2000  # calls to time

LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
          "SITTING", "STANDING", "LAYING"]


# ------------------------------------------------------------------
# Step 0: Quantize (produces int8 ONNX if not already present)
# ------------------------------------------------------------------

def maybe_quantize():
    if os.path.exists(ONNX_INT8_PATH):
        print(f"[quantize] {ONNX_INT8_PATH} already exists, skipping.")
        return

    # Step 1: ORT preprocessing — fixes shape inference issues that cause
    # quantize_dynamic to fail on raw torch.onnx.export output.
    # Adds shape annotations the quantizer needs to work correctly.
    if not os.path.exists(ONNX_PREP_PATH):
        print("[quantize] Running ORT preprocessing (shape annotation) ...")
        from onnxruntime.quantization import shape_inference as ort_shape
        ort_shape.quant_pre_process(
            input_model_path=ONNX_FLOAT32_PATH,
            output_model_path=ONNX_PREP_PATH,
            skip_optimization=False,
        )
        print(f"[quantize] Preprocessed → {ONNX_PREP_PATH}")
    else:
        print(f"[quantize] {ONNX_PREP_PATH} already exists, skipping preprocess.")

    # Step 2: Quantize from the preprocessed model, not the raw export
    print("[quantize] Running dynamic PTQ float32 → int8 ...")
    quantize_dynamic(
        model_input=ONNX_PREP_PATH,   # <-- preprocessed, not raw
        model_output=ONNX_INT8_PATH,
        weight_type=QuantType.QInt8,
    )
    print(f"[quantize] Saved → {ONNX_INT8_PATH}")


# ------------------------------------------------------------------
# Step 1: Load data + scaler
# ------------------------------------------------------------------

def load_test_data():
    scaler = joblib.load(SCALER_PATH)

    X = np.load(X_TEST_PATH).astype(np.float32)
    y = np.load(Y_TEST_PATH).squeeze().astype(int)
    y = y - 1  # match training: 1-indexed → 0-indexed

    X_scaled = scaler.transform(X).astype(np.float32)
    return X_scaled, y


# ------------------------------------------------------------------
# Step 2: Timing helper
# ------------------------------------------------------------------

def time_single_inference(fn, X: np.ndarray):
    """
    Runs fn(X[i]) N_WARMUP + N_RUNS times.
    Returns latency array (ms) for the N_RUNS timed calls.
    Each call uses a single sample — mimics real-time streaming inference.
    """
    latencies = []

    for i in range(N_WARMUP + N_RUNS):
        sample = X[i % len(X)][np.newaxis, :]  # shape (1, F)
        t0 = time.perf_counter()
        fn(sample)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if i >= N_WARMUP:
            latencies.append(elapsed_ms)

    return np.array(latencies)


def latency_stats(arr: np.ndarray) -> dict:
    return {
        "mean_ms": round(float(np.mean(arr)), 4),
        "p50_ms":  round(float(np.percentile(arr, 50)), 4),
        "p95_ms":  round(float(np.percentile(arr, 95)), 4),
        "p99_ms":  round(float(np.percentile(arr, 99)), 4),
        "throughput_per_sec": round(1000 / float(np.mean(arr)), 1),
    }


# ------------------------------------------------------------------
# Step 3: Per-runtime inference functions + accuracy
# ------------------------------------------------------------------

# --- PyTorch ---

def build_pytorch_fn(pth_path, input_dim):
    model = MLP(input_dim=input_dim, hidden_dim=128, num_classes=6)
    model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    model.eval()

    def fn(x: np.ndarray):
        with torch.no_grad():
            t = torch.from_numpy(x)
            return model(t).numpy()

    return fn


def pytorch_predict_all(fn, X):
    preds = []
    for i in range(len(X)):
        logits = fn(X[i][np.newaxis, :])
        preds.append(int(np.argmax(logits, axis=1)[0]))
    return np.array(preds)


# --- ONNX (shared for float32 and int8) ---

def build_onnx_fn(onnx_path):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options=sess_options)
    input_name = session.get_inputs()[0].name

    def fn(x: np.ndarray):
        return session.run(None, {input_name: x})[0]

    return fn, session


def onnx_predict_all(fn, X):
    preds = []
    for i in range(len(X)):
        out = fn(X[i][np.newaxis, :])
        preds.append(int(np.argmax(out, axis=1)[0]))
    return np.array(preds)


# --- XGBoost native ---

def build_xgb_fn(json_path):
    model = xgb.XGBClassifier()
    model.load_model(json_path)

    def fn(x: np.ndarray):
        # predict() returns int labels directly — no argmax needed
        return model.predict(x)

    return fn


def xgb_predict_all(fn, X):
    preds = []
    for i in range(len(X)):
        out = fn(X[i][np.newaxis, :])
        preds.append(int(out[0]))
    return np.array(preds)


# --- XGBoost ONNX ---
# onnxmltools XGBoost export produces two outputs:
#   output[0] → predicted label  (int64, shape (batch,))
#   output[1] → probabilities    (dict/map, shape (batch, n_classes))
# We use output[0] directly — no argmax needed, unlike the MLP ONNX.

def build_xgb_onnx_fn(onnx_path):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options=sess_options)
    input_name = session.get_inputs()[0].name

    def fn(x: np.ndarray):
        # output[0] is label, output[1] is prob map — we only need label
        return session.run(None, {input_name: x})[0]

    return fn


def xgb_onnx_predict_all(fn, X):
    preds = []
    for i in range(len(X)):
        out = fn(X[i][np.newaxis, :])
        preds.append(int(out[0]))
    return np.array(preds)


# ------------------------------------------------------------------
# Step 4: Run everything
# ------------------------------------------------------------------

def file_size_kb(path: str) -> float:
    total = os.path.getsize(path)
    ext = path + ".data"
    if os.path.exists(ext):
        total += os.path.getsize(ext)
    return round(total / 1024, 1)


def run_benchmark():
    maybe_quantize()

    X_test, y_test = load_test_data()
    input_dim = X_test.shape[1]
    print(f"\nTest set: {len(X_test)} samples, {input_dim} features\n")

    results = {}

    # ---- 1. PyTorch ----
    print("--- PyTorch native ---")
    pt_fn = build_pytorch_fn(PTH_PATH, input_dim)
    pt_lats = time_single_inference(pt_fn, X_test)
    pt_preds = pytorch_predict_all(pt_fn, X_test)
    pt_f1 = f1_score(y_test, pt_preds, average="macro")

    results["pytorch"] = {
        "file": PTH_PATH,
        "size_kb": file_size_kb(PTH_PATH),
        **latency_stats(pt_lats),
        "macro_f1": round(pt_f1, 4),
    }
    print(f"  size={results['pytorch']['size_kb']} KB  "
          f"p99={results['pytorch']['p99_ms']} ms  "
          f"macro_f1={pt_f1:.4f}")

    # ---- 2. ONNX float32 ----
    print("--- ONNX float32 ---")
    onnx_fn, _ = build_onnx_fn(ONNX_FLOAT32_PATH)
    onnx_lats = time_single_inference(onnx_fn, X_test)
    onnx_preds = onnx_predict_all(onnx_fn, X_test)
    onnx_f1 = f1_score(y_test, onnx_preds, average="macro")

    results["onnx_float32"] = {
        "file": ONNX_FLOAT32_PATH,
        "size_kb": file_size_kb(ONNX_FLOAT32_PATH),
        **latency_stats(onnx_lats),
        "macro_f1": round(onnx_f1, 4),
    }
    print(f"  size={results['onnx_float32']['size_kb']} KB  "
          f"p99={results['onnx_float32']['p99_ms']} ms  "
          f"macro_f1={onnx_f1:.4f}")

    # ---- 3. ONNX int8 ----
    print("--- ONNX int8 PTQ ---")
    int8_fn, _ = build_onnx_fn(ONNX_INT8_PATH)
    int8_lats = time_single_inference(int8_fn, X_test)
    int8_preds = onnx_predict_all(int8_fn, X_test)
    int8_f1 = f1_score(y_test, int8_preds, average="macro")

    results["onnx_int8"] = {
        "file": ONNX_INT8_PATH,
        "size_kb": file_size_kb(ONNX_INT8_PATH),
        **latency_stats(int8_lats),
        "macro_f1": round(int8_f1, 4),
    }
    print(f"  size={results['onnx_int8']['size_kb']} KB  "
          f"p99={results['onnx_int8']['p99_ms']} ms  "
          f"macro_f1={int8_f1:.4f}")

    # ---- 4. XGBoost native ----
    print("--- XGBoost native (.json) ---")
    xgb_fn = build_xgb_fn(XGB_JSON_PATH)
    xgb_lats = time_single_inference(xgb_fn, X_test)
    xgb_preds = xgb_predict_all(xgb_fn, X_test)
    xgb_f1 = f1_score(y_test, xgb_preds, average="macro")

    results["xgb_native"] = {
        "file": XGB_JSON_PATH,
        "size_kb": file_size_kb(XGB_JSON_PATH),
        **latency_stats(xgb_lats),
        "macro_f1": round(xgb_f1, 4),
    }
    print(f"  size={results['xgb_native']['size_kb']} KB  "
          f"p99={results['xgb_native']['p99_ms']} ms  "
          f"macro_f1={xgb_f1:.4f}")

    # ---- 5. XGBoost ONNX ----
    print("--- XGBoost ONNX ---")
    xgb_onnx_fn = build_xgb_onnx_fn(XGB_ONNX_PATH)
    xgb_onnx_lats = time_single_inference(xgb_onnx_fn, X_test)
    xgb_onnx_preds = xgb_onnx_predict_all(xgb_onnx_fn, X_test)
    xgb_onnx_f1 = f1_score(y_test, xgb_onnx_preds, average="macro")

    results["xgb_onnx"] = {
        "file": XGB_ONNX_PATH,
        "size_kb": file_size_kb(XGB_ONNX_PATH),
        **latency_stats(xgb_onnx_lats),
        "macro_f1": round(xgb_onnx_f1, 4),
    }
    print(f"  size={results['xgb_onnx']['size_kb']} KB  "
          f"p99={results['xgb_onnx']['p99_ms']} ms  "
          f"macro_f1={xgb_onnx_f1:.4f}")

    return results, y_test, pt_preds, onnx_preds, int8_preds, xgb_preds, xgb_onnx_preds


# ------------------------------------------------------------------
# Step 5: Format output
# ------------------------------------------------------------------

def format_table(results: dict) -> str:
    header = f"{'Runtime':<20} {'Size (KB)':>10} {'p50 (ms)':>10} {'p95 (ms)':>10} {'p99 (ms)':>10} {'Throughput/s':>14} {'Macro F1':>10}"
    sep = "-" * len(header)
    rows = []
    for name, r in results.items():
        rows.append(
            f"{name:<20} {r['size_kb']:>10} {r['p50_ms']:>10} "
            f"{r['p95_ms']:>10} {r['p99_ms']:>10} "
            f"{r['throughput_per_sec']:>14} {r['macro_f1']:>10}"
        )
    return "\n".join([header, sep] + rows)


def format_delta(results: dict) -> str:
    base = results["pytorch"]
    lines = ["\nDelta vs PyTorch baseline:"]
    for name in ["onnx_float32", "onnx_int8", "xgb_native", "xgb_onnx"]:
        r = results[name]
        size_delta  = round(r["size_kb"]  - base["size_kb"],  1)
        p99_delta   = round(r["p99_ms"]   - base["p99_ms"],   4)
        f1_delta    = round(r["macro_f1"] - base["macro_f1"], 4)
        lines.append(
            f"  {name:<20}: size {size_delta:+.1f} KB | "
            f"p99 {p99_delta:+.4f} ms | "
            f"macro_f1 {f1_delta:+.4f}"
        )

    # Also show XGBoost ONNX delta vs XGBoost native (runtime gain within same model)
    xgb_base = results["xgb_native"]
    xgb_onnx = results["xgb_onnx"]
    lines.append(
        f"\n  xgb_onnx vs xgb_native : "
        f"p99 {round(xgb_onnx['p99_ms'] - xgb_base['p99_ms'], 4):+.4f} ms | "
        f"macro_f1 {round(xgb_onnx['macro_f1'] - xgb_base['macro_f1'], 4):+.4f}"
    )
    return "\n".join(lines)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    results, y_test, pt_preds, onnx_preds, int8_preds, xgb_preds, xgb_onnx_preds = run_benchmark()

    table = format_table(results)
    delta = format_delta(results)

    print("\n" + "=" * 80)
    print(table)
    print(delta)
    print("=" * 80)

    # Per-class breakdown for the two most constrained runtimes
    print("\nPer-class F1 (ONNX int8) — checks MLP quantization didn't hurt specific classes:")
    print(classification_report(y_test, int8_preds, target_names=LABELS, digits=3))

    print("\nPer-class F1 (XGBoost ONNX) — compare tree vs MLP error patterns:")
    print(classification_report(y_test, xgb_onnx_preds, target_names=LABELS, digits=3))

    # Save
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    txt_output = table + "\n" + delta
    with open("benchmark_results.txt", "w") as f:
        f.write(txt_output)

    print("\nSaved → benchmark_results.json, benchmark_results.txt")
