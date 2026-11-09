#!/usr/bin/env bash
# run_all.sh: everything after data preparation, from scratch, in a fresh virtualenv.
#
#   bash scripts/run_all.sh [--with-loadtest] [--force-prepare] [--dry-run]
#
# Run it from the repo root. It:
#   1. creates .venv and installs requirements.txt
#   2. pins scikit-learn in requirements-serve.txt to the version it trains with
#   3. runs the unit tests
#   4. uses data/processed/*.npy if present (otherwise builds it from the raw UCI files)
#   5. trains the MLP and XGBoost, evaluates, benchmarks, makes the EDA plots
#   6. (--with-loadtest) runs the load-test sweep, about 12 minutes
#   7. fills the README tables from the result files
#
# Options
#   --with-loadtest   also run the load sweep. This REPLACES the load table in the README,
#                     so update the "Findings" bullets to match afterwards.
#   --force-prepare   rebuild data/processed from raw even if it exists
#   --dry-run         print the commands without running them
#
# Environment: UCI_DIR (default "data/raw/UCI HAR Dataset"), STREAM (default data/processed/train_stream.npy)
set -euo pipefail

LOADTEST=0; FORCE=0; DRY=0
for a in "$@"; do
  case "$a" in
    --with-loadtest) LOADTEST=1 ;;
    --force-prepare) FORCE=1 ;;
    --dry-run)       DRY=1 ;;
    -h|--help)       sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

UCI_DIR="${UCI_DIR:-data/raw/UCI HAR Dataset}"
STREAM="${STREAM:-data/processed/train_stream.npy}"
PY=".venv/bin/python"

[ -f inference/api/main.py ] || { echo "Run this from the repo root." >&2; exit 1; }

run() { echo; echo "+ $*"; if [ "$DRY" = 0 ]; then "$@"; fi; }
step() { echo; echo "=================== $* ==================="; }

step "1/7  virtualenv + dependencies"
run python3 -m venv .venv
run $PY -m pip install -q --upgrade pip
run $PY -m pip install -r requirements.txt

step "2/7  pin scikit-learn for the Docker image"
if [ "$DRY" = 0 ]; then
  $PY - <<'PYEOF'
import re, sklearn
p = "requirements-serve.txt"
s = open(p).read()
s = re.sub(r"(?m)^scikit-learn.*$", f"scikit-learn=={sklearn.__version__}", s)
open(p, "w").write(s)
print("requirements-serve.txt -> scikit-learn==" + sklearn.__version__)
PYEOF
else echo "+ (would pin scikit-learn in requirements-serve.txt)"; fi

step "3/7  unit tests"
run $PY tests/test_buffer.py
run $PY tests/test_metrics.py

step "4/7  data"
NEED_PREP=$FORCE
for f in X_train y_train X_test y_test train_stream; do
  [ -f "data/processed/$f.npy" ] || NEED_PREP=1
done
if [ "$NEED_PREP" = 1 ]; then
  [ -d "$UCI_DIR" ] || { echo "data/processed is incomplete and $UCI_DIR not found. Download UCI HAR into data/raw/ first." >&2; [ "$DRY" = 1 ] || exit 1; }
  run $PY scripts/prepare_data.py --uci "$UCI_DIR"
else
  echo "data/processed already complete, skipping preparation."
fi

step "5/7  train, evaluate, benchmark, EDA"
run mkdir -p model docs
run $PY train_mlp.py
run $PY train_xgboost.py
run $PY evaluate.py
run $PY benchmark.py
if [ -d "$UCI_DIR" ]; then
  run $PY eda.py --uci "$UCI_DIR" --split train
else
  echo "Skipping EDA: $UCI_DIR not found (eda.py needs the raw signals)."
fi

step "6/7  load test"
# Never let a stale results file overwrite the README's load table by accident.
if [ -f load_test_results.json ] && [ "$DRY" = 0 ]; then
  mv load_test_results.json load_test_results.previous.json
  echo "Moved the old load_test_results.json to load_test_results.previous.json"
fi
if [ "$LOADTEST" = 1 ]; then
  run $PY scripts/sweep.py 40:1 60:1 80:1 100:1 40:50 200:50 400:50 800:50 1600:50 --stream "$STREAM" --duration 60
else
  echo "Skipped (add --with-loadtest to run it). The README keeps its existing load table."
fi

step "7/7  fill README tables"
run $PY scripts/fill_readme.py

if [ "$DRY" = 0 ]; then
  step "check outputs"
  missing=0
  for f in model/mlp.pth model/mlp_model.onnx model/scaler.pkl model/xgb_model.json model/xgb_model.onnx \
           evaluation_results.json benchmark_results.json \
           docs/eda_signals.png docs/eda_features.png docs/confusion_matrices.png; do
    if [ -f "$f" ]; then echo "ok       $f"; else echo "MISSING  $f"; missing=1; fi
  done
  echo
  echo "README placeholders still to fill by hand:"; grep -n "TODO" README.md || echo "  none"
  [ "$missing" = 0 ] || { echo "Some outputs are missing; see the messages above." >&2; exit 1; }
  echo; echo "Done. Next: docker build -t har-service . && docker run --rm -p 8000:8000 har-service"
fi
