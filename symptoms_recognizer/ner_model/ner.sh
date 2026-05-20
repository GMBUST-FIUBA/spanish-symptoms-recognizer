#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

output_dir="$SCRIPT_DIR/output/model"
mkdir -p "$output_dir"

python "$SCRIPT_DIR/ner/run_ner.py" \
  --model_name_or_path base_model/bsc-bio-ehr-es \
  --train_file data/train_set.jsonl \
  --validation_file data/validation_set.jsonl \
  --test_file data/test_set.jsonl \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --num_train_epochs 5 \
  --load_best_model_at_end True \
  --metric_for_best_model f1 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --overwrite_output_dir \
  --fp16 True \
  --output_dir "$output_dir" 2>&1 | tee "$output_dir/train.log"