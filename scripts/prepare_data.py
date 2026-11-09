"""
prepare_data.py
---------------
Turns the raw UCI HAR download into every array the rest of the repo reads.

    python scripts/prepare_data.py --uci "data/raw/UCI HAR Dataset"

Writes to data/processed/ (default):
    X_train.npy, X_test.npy    (N, 13) handcrafted features, from utils.features.extract_features,
                               the SAME function the live service runs, so training and serving
                               cannot drift apart
    y_train.npy, y_test.npy    (N,) labels, 1-indexed as in the UCI files (the training and
                               evaluation scripts subtract 1)
    train_stream.npy           (T, 6) continuous signal for the load test / simulator

Channel order is total_acc x,y,z then body_gyro x,y,z, matching the service's
[ax, ay, az, gx, gy, gz].

train_stream.npy is rebuilt from the windows: UCI windows overlap by 50%, so the first
64 samples of each consecutive window tile the original signal. It is only used to replay
realistic-looking data into the server, so it is not important that segment boundaries
between activities are discontinuous.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.features import extract_features  # noqa: E402

SIGNALS = ("total_acc", "body_gyro")
AXES = "xyz"


def load_split(uci_dir, split):
    sig_dir = os.path.join(uci_dir, split, "Inertial Signals")
    if not os.path.isdir(sig_dir):
        sys.exit(f"Not found: {sig_dir}\n"
                 "Point --uci at the folder that contains train/ and test/ "
                 "(if the download is a zip inside a zip, unzip both).")
    chans = [np.loadtxt(os.path.join(sig_dir, f"{s}_{a}_{split}.txt"))
             for s in SIGNALS for a in AXES]
    windows = np.stack(chans, axis=-1).astype(np.float32)          # (N, 128, 6)
    y = np.loadtxt(os.path.join(uci_dir, split, f"y_{split}.txt")).astype(int)
    assert windows.shape[0] == y.shape[0], (windows.shape, y.shape)
    return windows, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uci", default="data/raw/UCI HAR Dataset")
    p.add_argument("--out", default="data/processed")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for split in ("train", "test"):
        windows, y = load_split(args.uci, split)
        X = np.vstack([extract_features(w) for w in windows]).astype(np.float32)
        np.save(os.path.join(args.out, f"X_{split}.npy"), X)
        np.save(os.path.join(args.out, f"y_{split}.npy"), y)
        print(f"{split}: windows {windows.shape} -> features {X.shape}, "
              f"class counts {np.bincount(y)[1:].tolist()}")

        if split == "train":
            stream = np.concatenate([w[:64] for w in windows], axis=0)
            np.save(os.path.join(args.out, "train_stream.npy"), stream)
            print(f"train_stream: {stream.shape}")

    print(f"Saved to {args.out}/")


if __name__ == "__main__":
    main()
