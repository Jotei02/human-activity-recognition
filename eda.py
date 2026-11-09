"""
eda.py
------
Three plots, nothing more:

  1. docs/eda_signals.png          raw 6-channel signal, one example window per class
  2. docs/eda_features.png         distribution of each of the 13 features, by class
  3. docs/confusion_matrices.png   confusion matrices (MLP and XGBoost), from evaluate.py

Usage
    python eda.py --uci "data/raw/UCI HAR Dataset" --split train
    python eda.py --uci "data/raw/UCI HAR Dataset" --split train --seed 1   # other example windows

Plot 3 reads evaluation_results.json, so run `python evaluate.py` first
(plots 1 and 2 work without it).
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.features import extract_features

LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
          "SITTING", "STANDING", "LAYING"]
SHORT = ["WLK", "W_UP", "W_DN", "SIT", "STD", "LAY"]
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
FEATURE_NAMES = ([f"mean_{c}" for c in CHANNELS]
                 + [f"std_{c}" for c in CHANNELS]
                 + ["rms_mag"])

# Colour-blind-safe categorical palette (Okabe-Ito). Dynamic classes are warm,
# static classes are cool, so the dynamic/static split reads at a glance.
COLORS = ["#D55E00", "#E69F00", "#CC79A7", "#0072B2", "#56B4E9", "#009E73"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 9,
})


def load_uci(uci_dir, split):
    """Windows (N, 128, 6) = total_acc xyz + body_gyro xyz, plus 0-indexed labels.
    Same channel order as scripts/prepare_labels.py and the live service."""
    sig_dir = os.path.join(uci_dir, split, "Inertial Signals")
    chans = []
    for sig in ("total_acc", "body_gyro"):
        for ax in "xyz":
            chans.append(np.loadtxt(os.path.join(sig_dir, f"{sig}_{ax}_{split}.txt")))
    X = np.stack(chans, axis=-1)                       # (N, 128, 6)
    y = np.loadtxt(os.path.join(uci_dir, split, f"y_{split}.txt")).astype(int) - 1
    return X, y


def plot_signals(X, y, out, seed=0):
    """6 classes (rows) x 6 channels (cols); same y-limits per column so
    amplitude differences between classes are visible, not hidden by autoscale."""
    rng = np.random.default_rng(seed)
    t = np.arange(X.shape[1]) / 50.0                   # 50 Hz
    picks = [rng.choice(np.where(y == c)[0]) for c in range(6)]

    fig, axes = plt.subplots(6, 6, figsize=(14, 8), sharex=True, sharey="col")
    for r, c in enumerate(range(6)):
        for ch in range(6):
            ax = axes[r, ch]
            ax.plot(t, X[picks[r], :, ch], color=COLORS[c], lw=1)
            if r == 0:
                ax.set_title(CHANNELS[ch])
            if ch == 0:
                ax.set_ylabel(SHORT[c], rotation=0, labelpad=22, va="center",
                              color=COLORS[c], fontweight="bold")
            if r == 5:
                ax.set_xlabel("s")
    fig.suptitle("Raw IMU window (2.56 s @ 50 Hz), one example per activity. "
                 "Columns share a y-axis.", y=1.0)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_features(X, y, out):
    """Box plot per feature, split by class. This is the plot that explains the
    SITTING/STANDING overlap: compare their boxes on every feature."""
    F = np.vstack([extract_features(w) for w in X])    # (N, 13)

    fig, axes = plt.subplots(3, 5, figsize=(15, 8))
    for k, ax in enumerate(axes.ravel()):
        if k >= F.shape[1]:
            ax.axis("off")
            continue
        data = [F[y == c, k] for c in range(6)]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                        medianprops={"color": "black"}, widths=0.65)
        for patch, col in zip(bp["boxes"], COLORS):
            patch.set_facecolor(col)
            patch.set_alpha(0.8)
        ax.set_xticks(range(1, 7))
        ax.set_xticklabels(SHORT, rotation=60, fontsize=7)
        ax.set_title(FEATURE_NAMES[k])
    fig.suptitle("Handcrafted feature distributions by class (outliers hidden)", y=1.0)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    # Quantify the overlap so the README claim isn't just eyeballed
    sit, std = F[y == 3], F[y == 4]
    pooled = np.sqrt((sit.var(axis=0) + std.var(axis=0)) / 2) + 1e-12
    d = np.abs(sit.mean(axis=0) - std.mean(axis=0)) / pooled
    print("\nSITTING vs STANDING separation per feature (Cohen's d; <0.5 = heavy overlap):")
    for name, v in sorted(zip(FEATURE_NAMES, d), key=lambda p: -p[1]):
        print(f"  {name:<10} d={v:.2f}")


def plot_confusion(results_json, out):
    with open(results_json) as f:
        res = json.load(f)
    models = [("MLP (PyTorch)", res["mlp"]), ("XGBoost", res["xgboost"])]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, r) in zip(axes, models):
        cm = np.array(r["confusion_matrix"])
        norm = cm / cm.sum(axis=1, keepdims=True)      # row-normalised: per-class recall
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.55 else "black")
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(SHORT); ax.set_yticklabels(SHORT)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{name}  (acc {r['accuracy']:.3f}, macro-F1 {r['macro_f1']:.3f})")
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Row-normalised (recall)")
    fig.savefig(out)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uci", default="data/raw/UCI HAR Dataset")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--eval-json", default="evaluation_results.json")
    p.add_argument("--out-dir", default="docs")
    p.add_argument("--seed", type=int, default=0, help="which example windows plot 1 shows")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    X, y = load_uci(args.uci, args.split)
    print(f"Loaded {args.split}: windows={X.shape}, class counts={np.bincount(y).tolist()}")

    plot_signals(X, y, f"{args.out_dir}/eda_signals.png", seed=args.seed)
    plot_features(X, y, f"{args.out_dir}/eda_features.png")
    if os.path.exists(args.eval_json):
        plot_confusion(args.eval_json, f"{args.out_dir}/confusion_matrices.png")
    else:
        print(f"{args.eval_json} not found; run evaluate.py first for the confusion matrix.")
    print(f"Saved plots -> {args.out_dir}/")


if __name__ == "__main__":
    main()
