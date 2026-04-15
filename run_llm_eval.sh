#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

RESULTS_DIR="$ROOT/llm_log"
mkdir -p "$RESULTS_DIR"

# ── 配置 ──────────────────────────────────────────────────────────────────────
# 要测试的模型列表
MODELS=(
     "gpt-5.2"
#    "claude-sonnet-4-6"
#    "grok-4-1-fast-reasoning"
#    "gemini-3.1-pro-preview"
)

# 定义各个数据集的路径
PI_DATA="$ROOT/data/prompt-injections/full_test.csv"
JC_BALANCED_DATA="$ROOT/data/jailbreak-classification-balanced/full_test.csv"
JC_IMBALANCED_DATA="$ROOT/data/jailbreak-classification-imbalanced/full_test.csv"
SG_DATA="$ROOT/data/safe-guard-prompt-injection/full_test.csv"

# 定义任务顺序（任务名:数据路径）
TASKS=(
    "PI:$PI_DATA"
    "JC_balanced:$JC_BALANCED_DATA"
    "JC_imbalanced:$JC_IMBALANCED_DATA"
    "SG:$SG_DATA"
)

# 每次 LLM 请求后的间隔秒数（0 表示不限速）
REQUEST_DELAY=0
# ─────────────────────────────────────────────────────────────────────────────

for MODEL in "${MODELS[@]}"; do
    for TASK_ENTRY in "${TASKS[@]}"; do
        TASK="${TASK_ENTRY%%:*}"
        DATA_PATH="${TASK_ENTRY#*:}"

        # 决定传给评测脚本的任务名，以及输出目录名中的任务名
        if [[ "$TASK" == "JC_balanced" ]] || [[ "$TASK" == "JC_imbalanced" ]]; then
            EVAL_TASK_NAME="JC"
            OUTPUT_TASK_NAME="JC"
        else
            EVAL_TASK_NAME="$TASK"
            OUTPUT_TASK_NAME="$TASK"
        fi

        OUTPUT="$RESULTS_DIR/${OUTPUT_TASK_NAME}_${MODEL//\//_}"

        # 对于 JC 的两个变体，需要在输出目录名中区分，避免覆盖
        if [[ "$TASK" == "JC_balanced" ]]; then
            OUTPUT="$RESULTS_DIR/JC_balanced_${MODEL//\//_}"
        elif [[ "$TASK" == "JC_imbalanced" ]]; then
            OUTPUT="$RESULTS_DIR/JC_imbalanced_${MODEL//\//_}"
        fi

        if [[ "$MODEL" == "claude-sonnet-4-6" && "$TASK" == "PI" ]]; then
            echo "跳过已完成评测: 任务=$TASK | 模型=$MODEL"
            echo ""
            continue
        fi

        echo "================================================================"
        echo "  任务: $TASK  |  模型: $MODEL"
        echo "  数据: $DATA_PATH"
        echo "  输出: $OUTPUT"
        echo "================================================================"

        python "$ROOT/src/eval_llm.py" \
            --task_name "$EVAL_TASK_NAME" \
            --data_path "$DATA_PATH" \
            --model "$MODEL" \
            --request_delay "$REQUEST_DELAY" \
            --output_path "$OUTPUT"

        echo ""
    done
done


#python "$ROOT/src/eval_llm.py" \
#    --task_name JC \
#    --data_path /home/han/llh/EvoShield/data/jailbreak-classification-balanced/full_test.csv \
#    --model grok-4.1 \
#    --request_delay "$REQUEST_DELAY" \
#    --output_path /home/han/llh/EvoShield/llm_log/JC_balanced_grok-4.1 \
#    --resume_from /home/han/llh/EvoShield/llm_log/JC_balanced_grok-4.1

echo "全部评测完成，结果保存在 $RESULTS_DIR"
