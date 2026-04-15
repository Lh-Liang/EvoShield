#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

#: "${MODEL_PATH:?Set MODEL_PATH to your pretrained model directory}"
#: "${TRAIN_DATA_PATH:?Set TRAIN_DATA_PATH to your training CSV path}"

#TASK_NAME="${TASK_NAME:-SG}"
#VAL_DATA_PATH="${VAL_DATA_PATH:-}"

#mkdir -p "$ROOT/ckpts" "$ROOT/log" "$ROOT/tensorboard"

#for rws in 0 16 32 48 80 96; do
#  echo "== review_window_size=${rws} =="
#  python "$ROOT/src/train.py" \
#    --task_name "$TASK_NAME" \
#    --model_path "$MODEL_PATH" \
#    --train_data_path "$TRAIN_DATA_PATH" \
#    ${VAL_DATA_PATH:+--val_data_path "$VAL_DATA_PATH"} \
#    --checkpoint_dir "$ROOT/ckpts" \
#    --log_dir "$ROOT/log" \
#    --tensorboard_dir "$ROOT/tensorboard" \
#    --review_window_size $rws \
#    "$@"
#done

#for rws in 0 16 32 48 80; do
#  echo "== review_window_size=${rws} =="
#  python "$ROOT/src/train.py" \
#    --review_window_size $rws \
#    "$@"
#done

#python "$ROOT/src/train.py" \
#    --model claude-sonnet-4-6 \
#    --task_name JC \
#    --train_data_path /home/zxh/code/EvoShield/data/jailbreak-classification-balanced/full_test.csv \
#    --review_window_size 128
#
#python "$ROOT/src/train.py" \
#    --model claude-sonnet-4-6 \
#    --task_name JC \
#    --train_data_path /home/zxh/code/EvoShield/data/jailbreak-classification-imbalanced/full_test.csv \
#    --review_window_size 128
#
#python "$ROOT/src/train.py" \
#    --model claude-sonnet-4-6 \
#    --task_name PI \
#    --train_data_path /home/zxh/code/EvoShield/data/prompt-injections/full_test.csv \
#    --review_window_size 128
#
#python "$ROOT/src/train.py" \
#    --model claude-sonnet-4-6 \
#    --task_name SG \
#    --train_data_path /home/zxh/code/EvoShield/data/safe-guard-prompt-injection/full_test.csv \
#    --review_window_size 128

CKPT_ROOT="$ROOT/ckpts"
TB_ROOT="$ROOT/tensorboard"
LOG_ROOT="$ROOT/log"

mkdir -p "$CKPT_ROOT" "$TB_ROOT" "$LOG_ROOT"

run_train() {
  local model="$1"
  local task_name="$2"
  local train_data_path="$3"
  local data_variant=""
  local run_name=""

  if [[ "$train_data_path" == *"jailbreak-classification-balanced"* ]]; then
    data_variant="balanced"
  elif [[ "$train_data_path" == *"jailbreak-classification-imbalanced"* ]]; then
    data_variant="imbalanced"
  fi

  if [[ -n "$data_variant" ]]; then
    run_name="${task_name}_${data_variant}_${model}"
  else
    run_name="${task_name}_${model}"
  fi

  local ckpt_dir="$CKPT_ROOT/$run_name"
  local tb_dir="$TB_ROOT/$run_name"
  local log_dir="$LOG_ROOT/$run_name"

  mkdir -p "$ckpt_dir" "$tb_dir" "$log_dir"

  echo "== model=${model}, task=${task_name}, run=${run_name} =="
  python "$ROOT/src/train.py" \
      --model "$model" \
      --task_name "$task_name" \
      --train_data_path "$train_data_path" \
      --checkpoint_dir "$ckpt_dir" \
      --tensorboard_dir "$tb_dir" \
      --log_dir "$log_dir" \
      --review_window_size 0
}

#run_train "gpt-5.2" "JC" "/home/han/llh/EvoShield/data/jailbreak-classification-balanced/full_test.csv"
#run_train "gpt-5.2" "JC" "/home/han/llh/EvoShield/data/jailbreak-classification-imbalanced/full_test.csv"
#run_train "gpt-5.2" "PI" "/home/han/llh/EvoShield/data/prompt-injections/full_test.csv"
#run_train "gpt-5.2" "SG" "/home/han/llh/EvoShield/data/safe-guard-prompt-injection/full_test.csv"

run_train "grok-4-1-fast-reasoning" "JC" "/home/han/llh/EvoShield/data/jailbreak-classification-balanced/full_test.csv"
run_train "grok-4-1-fast-reasoning" "JC" "/home/han/llh/EvoShield/data/jailbreak-classification-imbalanced/full_test.csv"
run_train "grok-4-1-fast-reasoning" "PI" "/home/han/llh/EvoShield/data/prompt-injections/full_test.csv"
run_train "grok-4-1-fast-reasoning" "SG" "/home/han/llh/EvoShield/data/safe-guard-prompt-injection/full_test.csv"
#
#run_train "gemini-3.1-pro-preview" "JC" "/home/han/llh/EvoShield/data/jailbreak-classification-balanced/full_test.csv"
#run_train "gemini-3.1-pro-preview" "JC" "/home/han/llh/EvoShield/data/jailbreak-classification-imbalanced/full_test.csv"
#run_train "gemini-3.1-pro-preview" "PI" "/home/han/llh/EvoShield/data/prompt-injections/full_test.csv"
#run_train "gemini-3.1-pro-preview" "SG" "/home/han/llh/EvoShield/data/safe-guard-prompt-injection/full_test.csv"



