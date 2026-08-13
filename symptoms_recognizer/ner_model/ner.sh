#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Directorios de salida para cada fase
PHASE1_DIR="$SCRIPT_DIR/output/model_phase1_synthetic"
PHASE2_DIR="$SCRIPT_DIR/output/model_phase2_natural"

mkdir -p "$PHASE1_DIR"
mkdir -p "$PHASE2_DIR"

echo "==========================================================="
echo " INICIANDO FASE 2: Fine-Tuning con datos NATURALES"
echo "==========================================================="

NATURAL_TRAIN_FILE="data/train_set.jsonl"
NATURAL_VAL_FILE="data/validation_set.jsonl"
NATURAL_TEST_FILE="data/test_set.jsonl"

python "$SCRIPT_DIR/ner/run_ner.py" \
  --model_name_or_path "base_model/bsc-bio-ehr-es" \
  --train_file "$NATURAL_TRAIN_FILE" \
  --validation_file "$NATURAL_VAL_FILE" \
  --test_file "$NATURAL_TEST_FILE" \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 10 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --num_train_epochs 5 \
  --load_best_model_at_end True \
  --metric_for_best_model f1 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 2 \
  --overwrite_output_dir \
  --fp16 True \
  --dataloader_num_workers 4 \
  --preprocessing_num_workers 8 \
  --output_dir "$PHASE2_DIR" 2>&1 | tee "$PHASE2_DIR/train_phase2.log"