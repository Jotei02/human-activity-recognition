"""
train_xgboost.py
----------------
Trains an XGBoost classifier on the HAR feature set and exports it two ways:
  1. Native XGBoost model  → model/xgb_model.json   (for benchmarking)
  2. ONNX                  → model/xgb_model.onnx   (for production / ORT inference)

The same scaler from MLP training is reused — do NOT refit it here,
or your feature distributions will differ between models.

Dependencies
    pip install xgboost scikit-learn onnxmltools skl2onnx
"""

import os
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

SCALER_PATH    = "model/scaler.pkl"
X_TRAIN_PATH   = "data/processed/X_train.npy"
Y_TRAIN_PATH   = "data/processed/y_train.npy"
X_TEST_PATH    = "data/processed/X_test.npy"
Y_TEST_PATH    = "data/processed/y_test.npy"

XGB_MODEL_PATH  = "model/xgb_model.json"
XGB_ONNX_PATH   = "model/xgb_model.onnx"

LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
          "SITTING", "STANDING", "LAYING"]

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------

os.makedirs("model", exist_ok=True)   # fresh clone has no model/ yet
print("Loading data...")

# Reuse the scaler fitted during MLP training — critical for fair comparison.
# Both models must see the same scaled feature space.
scaler = joblib.load(SCALER_PATH)

X_train_raw = np.load(X_TRAIN_PATH).astype(np.float32)
y_train     = np.load(Y_TRAIN_PATH).squeeze().astype(int) - 1  # 1-indexed → 0-indexed

X_test_raw  = np.load(X_TEST_PATH).astype(np.float32)
y_test      = np.load(Y_TEST_PATH).squeeze().astype(int) - 1

X_train = scaler.transform(X_train_raw).astype(np.float32)
X_test  = scaler.transform(X_test_raw).astype(np.float32)

print(f"  Train: {X_train.shape}  Test: {X_test.shape}")
print(f"  Classes: {np.unique(y_train)}")

# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------

print("\nTraining XGBoost...")

model = xgb.XGBClassifier(
    n_estimators=300,       # number of trees
    max_depth=6,            # tree depth — controls model complexity
    learning_rate=0.1,
    subsample=0.8,          # row subsampling per tree — reduces overfitting
    colsample_bytree=0.8,   # feature subsampling per tree
    use_label_encoder=False,
    eval_metric="mlogloss",
    objective="multi:softprob",
    num_class=6,
    n_jobs=-1,              # use all CPU cores
    random_state=42,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,             # print eval loss every 50 rounds
)

# ------------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------------

print("\nEvaluating on test set...")
y_pred = model.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average="macro")
print(f"\nMacro F1: {macro_f1:.4f}")
print("\nPer-class report:")
print(classification_report(y_test, y_pred, target_names=LABELS, digits=3))

# ------------------------------------------------------------------
# Save native XGBoost model (.json)
# ------------------------------------------------------------------

# .json is the modern XGBoost format — human-readable, version-stable.
# Avoid .pkl for XGBoost; json is safer across xgboost versions.
model.save_model(XGB_MODEL_PATH)
print(f"\nSaved XGBoost model → {XGB_MODEL_PATH}")

# ------------------------------------------------------------------
# Export to ONNX
# ------------------------------------------------------------------

print("Exporting to ONNX...")

# onnxmltools needs the internal booster, not the sklearn wrapper
booster = model.get_booster()

initial_type = [("input", FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_xgboost(booster, initial_types=initial_type)

with open(XGB_ONNX_PATH, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Saved ONNX model     → {XGB_ONNX_PATH}")

# ------------------------------------------------------------------
# Quick sanity check: ONNX output matches native output
# ------------------------------------------------------------------

# import onnxruntime as ort

# session = ort.InferenceSession(XGB_ONNX_PATH)
# input_name = session.get_inputs()[0].name

# # ONNX output for XGBoost is a dict with 'label' and 'probabilities'
# onnx_out = session.run(None, {input_name: X_test[:5]})

# native_preds = model.predict(X_test[:5])
# onnx_preds   = onnx_out[0]  # label array

# print("\nSanity check (first 5 samples):")
# print(f"  Native : {native_preds}")
# print(f"  ONNX   : {onnx_preds}")
# match = np.array_equal(native_preds, onnx_preds)
# print(f"  Match  : {match}")
# if not match:
#     print("  WARNING: mismatch — check onnxmltools version compatibility.")

# print("\nDone.")
