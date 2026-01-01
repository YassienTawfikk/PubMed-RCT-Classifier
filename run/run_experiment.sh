#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Project root is one level up from run/
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project Root: $PROJECT_ROOT"
echo "=========================================="
echo "Starting Unified Pipeline (Train -> Eval)"
echo "=========================================="

# 1. Training
echo "[1/2] Running Training..."
python3 "$PROJECT_ROOT/script/training.py" \
  --data.data_dir "$PROJECT_ROOT/pubmed_rct" \
  --data.batch_size 32 \
  --train.epochs 5 \
  --train.model_save_dir "$PROJECT_ROOT/classifier_core" \
  --model_cfg.embed_dim 128

if [ $? -ne 0 ]; then
    echo "Training failed!"
    exit 1
fi

# 2. Evaluation
echo "[2/2] Running Evaluation..."
python3 "$PROJECT_ROOT/script/evaluate.py" \
  --data.data_dir "$PROJECT_ROOT/pubmed_rct" \
  --model_path "$PROJECT_ROOT/classifier_core/transformer_model.keras" \
  --data.batch_size 32

if [ $? -ne 0 ]; then
    echo "Evaluation failed!"
    exit 1
fi

echo "=========================================="
echo "Pipeline Completed Successfully!"
echo "=========================================="
