#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

output_dir="$SCRIPT_DIR/output/model"
mkdir -p "$output_dir"

python "$SCRIPT_DIR/ner/run_ner.py" \
  --model_name_or_path _MODEL_NAME_ \
  --train_file _TRAIN_FILE_PATH_ \
  --validation_file _VALIDATION_FILE_PATH_ \
  --test_file _TEST_TILE_PATH_ \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 10 \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --overwrite_output_dir \
  --logging_dir "$output_dir/tb" \
  --output_dir "$output_dir" 2>&1 | tee "$output_dir/train.log"
