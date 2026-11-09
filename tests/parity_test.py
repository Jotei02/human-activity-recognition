"""
parity_test.py
--------------
Train/serve parity: does the serving path (SlidingWindow -> resample -> features ->
scaler -> ONNX) give the same predictions as the offline path (raw UCI window -> features
-> scaler -> ONNX)?

    python tests/parity_test.py --uci "data/raw/UCI HAR Dataset"

It uses the service's own components (SlidingWindow, extract_features, ONNXPredictor), not
HTTP, so it checks the data path, not the network layer.

For every UCI test window (128 x 6):
  offline : features of the raw window
  serving : the same 128 samples are fed to a fresh SlidingWindow with timestamps at
            exactly 50 Hz, then get_window() -> features
Reported: feature differences, label agreement between the two paths, and accuracy of each
path against the true labels. Exits 1 if agreement < 99% or accuracy differs by > 1 point.

Without model files it only compares features (no prediction part).
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from utils.buffer import SlidingWindow          # noqa: E402
from utils.features import extract_features     # noqa: E402
from scripts.prepare_data import load_split             # noqa: E402

LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]


def serving_window(window, period=0.02):
    """Feed a raw (128, 6) window through the service's buffer exactly as a 50 Hz stream."""
    buf = SlidingWindow(n_channels=6)
    end = time.time()
    ts = [end - (len(window) - 1 - k) * period for k in range(len(window))]
    buf.append_many(ts, window.tolist())
    return buf.get_window()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uci", default="data/raw/UCI HAR Dataset")
    p.add_argument("--model", default=os.path.join(ROOT, "model/mlp_model.onnx"))
    p.add_argument("--scaler", default=os.path.join(ROOT, "model/scaler.pkl"))
    p.add_argument("--limit", type=int, default=0, help="use only the first N windows")
    args = p.parse_args()

    windows, y = load_split(args.uci, "test")
    y = y - 1
    if args.limit:
        windows, y = windows[:args.limit], y[:args.limit]

    have_model = os.path.exists(args.model) and os.path.exists(args.scaler)
    if have_model:
        from utils.predictor import ONNXPredictor
        pred = ONNXPredictor(args.model, args.scaler)

    f_off, f_srv, p_off, p_srv, rejected = [], [], [], [], 0
    for w in windows:
        sw = serving_window(w)
        if sw is None:
            rejected += 1
            continue
        fo, fs = extract_features(w), extract_features(sw)
        f_off.append(fo[0]); f_srv.append(fs[0])
        if have_model:
            p_off.append(LABELS.index(pred.predict(fo)[0]))
            p_srv.append(LABELS.index(pred.predict(fs)[0]))

    f_off, f_srv = np.array(f_off), np.array(f_srv)
    n = len(f_off)
    print(f"windows: {len(windows)}, rejected by the service buffer: {rejected}, compared: {n}")
    scale = f_off.std(axis=0) + 1e-9
    rel = np.abs(f_srv - f_off).max(axis=0) / scale
    print("max |feature diff| / feature std, per feature:")
    print("  " + " ".join(f"{v:.3f}" for v in rel))

    ok = rejected == 0
    if have_model:
        p_off, p_srv, yy = np.array(p_off), np.array(p_srv), y[:len(p_off)] if rejected == 0 else None
        agree = float((p_off == p_srv).mean())
        print(f"label agreement offline vs serving: {agree:.4f} ({int((p_off != p_srv).sum())} of {n} differ)")
        if yy is not None:
            a_off, a_srv = float((p_off == yy).mean()), float((p_srv == yy).mean())
            print(f"accuracy offline {a_off:.4f}  serving {a_srv:.4f}  (diff {a_srv - a_off:+.4f})")
            ok = ok and agree >= 0.99 and abs(a_srv - a_off) <= 0.01
        else:
            ok = ok and agree >= 0.99
    else:
        print("no model files found: features compared only")
    print("PARITY OK" if ok else "PARITY FAILED: investigate before claiming parity")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()