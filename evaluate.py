"""
evaluate.py
-----------
Evaluates two model families on the held-out test set:
  1. MLP (PyTorch native)
  2. XGBoost (native .json)

Per-model output
  - Accuracy
  - Per-class F1, precision, recall
  - Confusion matrix

Cross-model output
  - Agreement rate (% samples both models predict identically)
  - Samples where MLP wrong, XGBoost right  → XGBoost advantage
  - Samples where XGBoost wrong, MLP right  → MLP advantage
  - Samples where both are wrong            → hard samples

Usage
    python evaluate.py

Outputs
    evaluation_results.json     machine-readable summary
    confusion_mlp.txt           confusion matrix for MLP
    confusion_xgb.txt           confusion matrix for XGBoost
"""

import json
import numpy as np
import torch
import joblib
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from model import MLP

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

PTH_PATH      = "model/mlp.pth"
XGB_JSON_PATH = "model/xgb_model.json"
SCALER_PATH   = "model/scaler.pkl"
X_TEST_PATH   = "data/processed/X_test.npy"
Y_TEST_PATH   = "data/processed/y_test.npy"

LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
          "SITTING", "STANDING", "LAYING"]

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------

def load_test_data():
    scaler  = joblib.load(SCALER_PATH)
    X       = np.load(X_TEST_PATH).astype(np.float32)
    y       = np.load(Y_TEST_PATH).squeeze().astype(int) - 1  # 0-indexed
    X_scaled = scaler.transform(X).astype(np.float32)
    return X_scaled, y

# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

def mlp_predict_all(X: np.ndarray, input_dim: int) -> np.ndarray:
    model = MLP(input_dim=input_dim, hidden_dim=128, num_classes=6)
    model.load_state_dict(torch.load(PTH_PATH, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
    return torch.argmax(logits, dim=1).numpy().astype(int)


def xgb_predict_all(X: np.ndarray) -> np.ndarray:
    model = xgb.XGBClassifier()
    model.load_model(XGB_JSON_PATH)
    return model.predict(X).astype(int)

# ------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------

def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Full classification report + confusion matrix for one model."""
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report   = classification_report(y_true, y_pred, target_names=LABELS,
                                     digits=3, output_dict=True)
    cm       = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {macro_f1:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=3))
    print("  Confusion matrix (rows=true, cols=pred):")
    print(f"  Labels: {LABELS}")
    print(_format_cm(cm))

    return {
        "accuracy":  round(acc, 4),
        "macro_f1":  round(macro_f1, 4),
        "per_class": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall":    round(report[label]["recall"],    4),
                "f1":        round(report[label]["f1-score"],  4),
                "support":   int(report[label]["support"]),
            }
            for label in LABELS
        },
        "confusion_matrix": cm.tolist(),
    }


def _format_cm(cm: np.ndarray) -> str:
    """Pretty-print a confusion matrix with label headers."""
    short = ["WLK", "W_UP", "W_DN", "SIT", "STD", "LAY"]
    col_w = 6
    header = " " * 6 + "".join(f"{s:>{col_w}}" for s in short)
    lines  = [header]
    for i, row in enumerate(cm):
        row_str = f"{short[i]:<6}" + "".join(f"{v:>{col_w}}" for v in row)
        lines.append(row_str)
    return "\n".join("  " + l for l in lines)

# ------------------------------------------------------------------
# Disagreement analysis
# ------------------------------------------------------------------

def disagreement_analysis(
    y_true:    np.ndarray,
    mlp_preds: np.ndarray,
    xgb_preds: np.ndarray,
) -> dict:
    """
    Splits the test set into four buckets:
      both_correct    — agreement, both right
      both_wrong      — hard samples, both fail
      mlp_only        — MLP right, XGBoost wrong
      xgb_only        — XGBoost right, MLP wrong

    For each error bucket, reports which (true_class → predicted_class)
    confusions are most common — this shows *where* each model has an edge.
    """
    mlp_correct = mlp_preds == y_true
    xgb_correct = xgb_preds == y_true

    both_correct = mlp_correct & xgb_correct
    both_wrong   = ~mlp_correct & ~xgb_correct
    mlp_only     = mlp_correct & ~xgb_correct   # MLP right, XGB wrong
    xgb_only     = ~mlp_correct & xgb_correct   # XGB right, MLP wrong

    n = len(y_true)

    print(f"\n{'='*60}")
    print("  Disagreement Analysis")
    print(f"{'='*60}")
    print(f"  Total test samples : {n}")
    print(f"  Both correct       : {both_correct.sum():>5}  ({both_correct.mean()*100:.1f}%)")
    print(f"  Both wrong         : {both_wrong.sum():>5}  ({both_wrong.mean()*100:.1f}%)")
    print(f"  MLP only correct   : {mlp_only.sum():>5}  ({mlp_only.mean()*100:.1f}%)")
    print(f"  XGBoost only correct: {xgb_only.sum():>5}  ({xgb_only.mean()*100:.1f}%)")

    def top_confusions(mask, pred_wrong, label=""):
        """
        For samples in `mask`, show the most common true→predicted pairs
        where `pred_wrong` is the model making the mistake.
        """
        if mask.sum() == 0:
            return {}
        true_c = y_true[mask]
        pred_c = pred_wrong[mask]
        pairs  = {}
        for t, p in zip(true_c, pred_c):
            key = f"{LABELS[t]} → {LABELS[p]}"
            pairs[key] = pairs.get(key, 0) + 1
        # sort by frequency
        pairs = dict(sorted(pairs.items(), key=lambda x: -x[1]))
        print(f"\n  {label} (top confusions):")
        for pair, count in list(pairs.items())[:5]:
            print(f"    {pair:<45} {count:>4} samples")
        return pairs

    print()
    xgb_wrong_confusions = top_confusions(
        mlp_only, xgb_preds,               # MLP right, XGB wrong → show XGB's errors here
        label="Where XGBoost fails but MLP succeeds (XGB errors)"
    )
    mlp_wrong_confusions = top_confusions(
        xgb_only, mlp_preds,               # XGB right, MLP wrong → show MLP's errors here
        label="Where MLP fails but XGBoost succeeds (MLP errors)"
    )
    hard_confusions = top_confusions(
        both_wrong, mlp_preds,
        label="Hard samples (both wrong) — MLP predicted"
    )

    return {
        "total": n,
        "both_correct":  {"count": int(both_correct.sum()), "pct": round(both_correct.mean()*100, 2)},
        "both_wrong":    {"count": int(both_wrong.sum()),   "pct": round(both_wrong.mean()*100,   2)},
        "mlp_only":      {"count": int(mlp_only.sum()),     "pct": round(mlp_only.mean()*100,     2)},
        "xgb_only":      {"count": int(xgb_only.sum()),     "pct": round(xgb_only.mean()*100,     2)},
        "xgb_errors_when_mlp_wins": xgb_wrong_confusions,
        "mlp_errors_when_xgb_wins": mlp_wrong_confusions,
        "hard_sample_confusions":   hard_confusions,
    }

# ------------------------------------------------------------------
# Per-class difficulty ranking
# ------------------------------------------------------------------

def class_difficulty(y_true, mlp_preds, xgb_preds):
    """
    Ranks classes by how often *both* models get them wrong.
    High both-wrong rate = genuinely hard class, not a model-specific weakness.
    """
    print(f"\n{'='*60}")
    print("  Per-class difficulty (both models wrong)")
    print(f"{'='*60}")
    print(f"  {'Class':<25} {'Support':>8} {'Both wrong':>12} {'Rate':>8}")
    print(f"  {'-'*55}")

    class_stats = {}
    for c, label in enumerate(LABELS):
        mask        = y_true == c
        support     = mask.sum()
        both_wrong  = ((mlp_preds != y_true) & (xgb_preds != y_true) & mask).sum()
        rate        = both_wrong / support if support > 0 else 0.0
        print(f"  {label:<25} {support:>8} {both_wrong:>12} {rate*100:>7.1f}%")
        class_stats[label] = {
            "support":    int(support),
            "both_wrong": int(both_wrong),
            "rate":       round(rate, 4),
        }

    return class_stats

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading test data...")
    X_test, y_test = load_test_data()
    input_dim = X_test.shape[1]
    print(f"  {len(X_test)} samples, {input_dim} features, {len(LABELS)} classes")

    print("\nRunning inference...")
    mlp_preds = mlp_predict_all(X_test, input_dim)
    xgb_preds = xgb_predict_all(X_test)

    # --- per-model evaluation ---
    mlp_results = evaluate_model("MLP (PyTorch)", y_test, mlp_preds)
    xgb_results = evaluate_model("XGBoost",       y_test, xgb_preds)

    # --- cross-model analysis ---
    disagree    = disagreement_analysis(y_test, mlp_preds, xgb_preds)
    difficulty  = class_difficulty(y_test, mlp_preds, xgb_preds)

    # --- save outputs ---
    summary = {
        "mlp":              mlp_results,
        "xgboost":          xgb_results,
        "disagreement":     disagree,
        "class_difficulty": difficulty,
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save confusion matrices as plain text for README
    def cm_to_text(name, cm):
        lines = [f"Confusion Matrix — {name}",
                 f"Labels: {LABELS}", ""]
        short = ["WLK", "W_UP", "W_DN", "SIT", "STD", "LAY"]
        lines.append("       " + "  ".join(f"{s:>5}" for s in short))
        for i, row in enumerate(cm):
            lines.append(f"{short[i]:<7}" + "  ".join(f"{v:>5}" for v in row))
        return "\n".join(lines)

    with open("confusion_mlp.txt", "w") as f:
        f.write(cm_to_text("MLP (PyTorch)", np.array(mlp_results["confusion_matrix"])))

    with open("confusion_xgb.txt", "w") as f:
        f.write(cm_to_text("XGBoost", np.array(xgb_results["confusion_matrix"])))

    print(f"\n{'='*60}")
    print("  Saved outputs:")
    print("    evaluation_results.json")
    print("    confusion_mlp.txt")
    print("    confusion_xgb.txt")
    print(f"{'='*60}")
