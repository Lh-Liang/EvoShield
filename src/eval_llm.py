"""
Pure LLM Evaluation Script for EvoShield
直接使用 LLM 对测试集进行预测，测试纯 LLM 的分类能力
用法:
    export PYTHONPATH=".:src"
    python src/eval_llm.py \
        --task_name SG \
        --data_path data/safe-guard-prompt-injection/full_test.csv \
        --model gpt-4o-mini \
        [--max_samples 200] \
        [--output_path llm_log/SG_gpt-4o-mini]
"""
import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 支持从 src/ 直接运行
sys.path.insert(0, str(Path(__file__).parent))

from llm_client import LLMClient
from prompt import TASK_CATEGORIES


# ── CSV 读取 ──────────────────────────────────────────────────────────────────

def load_pi(data_path: str):
    """PI: ArticleId, text, label"""
    examples = []
    with open(data_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            if len(row) < 3:
                continue
            _, text, label = row[0], row[1], row[2]
            try:
                examples.append((text, int(label)))
            except ValueError:
                continue
    return examples


def load_jc(data_path: str):
    """JC: prompt, type (benign/jailbreak)"""
    label_map = {"benign": 0, "jailbreak": 1}
    examples = []
    with open(data_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            if len(row) < 2:
                continue
            text, label_str = row[0], row[1]
            if label_str in label_map:
                examples.append((text, label_map[label_str]))
            else:
                try:
                    examples.append((text, int(label_str)))
                except ValueError:
                    continue
    return examples


def load_sg(data_path: str):
    """SG: ArticleId, text, label"""
    return load_pi(data_path)  # 相同格式


LOADERS = {"PI": load_pi, "JC": load_jc, "SG": load_sg}

PROGRESS_FILE_NAME = "progress.jsonl"
LOG_PROGRESS_PATTERN = re.compile(r"\[(\d+)/(\d+)\]\s+已完成\s+(\d+)\s+条.*跳过:\s+(\d+)")


def resolve_resume_progress_path(resume_from: str) -> Path:
    resume_path = Path(resume_from)
    if resume_path.is_dir():
        return resume_path / PROGRESS_FILE_NAME
    if resume_path.suffix == ".log":
        return resume_path.parent / PROGRESS_FILE_NAME
    if resume_path.name == PROGRESS_FILE_NAME:
        return resume_path
    raise ValueError(f"无法从 resume_from 推断进度文件路径: {resume_from}")


def parse_resume_index_from_log(log_path: Path) -> int:
    if not log_path.exists():
        raise FileNotFoundError(f"恢复日志不存在: {log_path}")

    last_index = 0
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            match = LOG_PROGRESS_PATTERN.search(line)
            if match:
                last_index = int(match.group(1))
    return last_index


def load_progress(progress_path: Path, task_name: str, model: str, data_path: str):
    records_by_index = {}

    with open(progress_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"进度文件第 {line_number} 行不是合法 JSON: {progress_path}") from exc

            record_task = record.get("task")
            record_model = record.get("model")
            record_data_path = record.get("data_path")
            if record_task != task_name or record_model != model or record_data_path != data_path:
                raise ValueError(
                    "进度文件与当前评测参数不匹配: "
                    f"task={record_task}, model={record_model}, data_path={record_data_path}"
                )

            index = record.get("index")
            label = record.get("label")
            pred = record.get("pred")
            skipped = record.get("skipped")
            if not isinstance(index, int) or not isinstance(label, int) or not isinstance(skipped, bool):
                raise ValueError(f"进度文件第 {line_number} 行字段不合法: {progress_path}")

            records_by_index[index] = {
                "label": label,
                "pred": pred,
                "skipped": skipped,
            }

    return records_by_index


def append_progress_record(progress_path: Path, task_name: str, model: str, data_path: str, index: int, label: int, pred):
    record = {
        "task": task_name,
        "model": model,
        "data_path": data_path,
        "index": index,
        "label": label,
        "pred": pred,
        "skipped": pred is None,
    }
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_metrics_from_records(records_by_index: dict):
    all_labels = []
    all_preds = []
    skipped = 0

    for index in sorted(records_by_index):
        record = records_by_index[index]
        if record["skipped"]:
            skipped += 1
            continue
        all_labels.append(record["label"])
        all_preds.append(record["pred"])

    return all_labels, all_preds, skipped


# ── 主评测逻辑 ─────────────────────────────────────────────────────────────────

def evaluate(args):
    task_name = args.task_name.upper()
    categories = TASK_CATEGORIES[task_name]
    out_dir = None
    run_timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # 加载数据
    loader = LOADERS[task_name]
    examples = loader(args.data_path)
    if not examples:
        logger.error("没有加载到任何样本，请检查数据路径和格式。")
        sys.exit(1)

    if args.max_samples and args.max_samples < len(examples):
        examples = examples[:args.max_samples]
        logger.info(f"使用前 {args.max_samples} 条样本进行评测")

    # 配置日志文件
    progress_path = None
    if args.output_path:
        out_dir = Path(args.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        progress_path = out_dir / PROGRESS_FILE_NAME
        log_path = out_dir / f"{run_timestamp}.log"
        logger.add(str(log_path))
        logger.info(f"日志将实时写入目录: {out_dir}")

    logger.info(f"任务: {task_name}  |  样本数: {len(examples)}  |  模型: {args.model}")
    logger.info(f"类别: {categories}")

    client = LLMClient(model=args.model)

    records_by_index = {}
    resume_source = None
    inferred_start_index = 0

    if args.resume_from:
        resume_path = Path(args.resume_from)
        candidate_progress_path = resolve_resume_progress_path(args.resume_from)
        if candidate_progress_path.exists():
            records_by_index = load_progress(candidate_progress_path, task_name, args.model, args.data_path)
            resume_source = str(candidate_progress_path)
            logger.info(f"从进度文件恢复: {candidate_progress_path}，已恢复 {len(records_by_index)} 条记录")
        elif resume_path.suffix == ".log":
            inferred_start_index = parse_resume_index_from_log(resume_path)
            resume_source = str(resume_path)
            logger.warning(
                f"未找到进度文件 {candidate_progress_path}，将仅根据日志从第 {inferred_start_index} 条后继续，"
                "此前结果不会计入最终指标"
            )
        else:
            raise FileNotFoundError(f"恢复来源不存在: {args.resume_from}")
    elif progress_path is not None and progress_path.exists():
        records_by_index = load_progress(progress_path, task_name, args.model, args.data_path)
        resume_source = str(progress_path)
        logger.info(f"检测到已有进度文件，自动恢复: {progress_path}，已恢复 {len(records_by_index)} 条记录")

    if records_by_index:
        inferred_start_index = max(records_by_index) + 1

    if inferred_start_index >= len(examples):
        logger.info("所有样本已完成，无需继续评测。")

    for i, (text, label) in enumerate(examples):
        if i in records_by_index or i < inferred_start_index:
            continue

        pred = client.predict(text, task_name=task_name, categories=categories)
        records_by_index[i] = {
            "label": label,
            "pred": pred,
            "skipped": pred is None,
        }

        if progress_path is not None:
            append_progress_record(progress_path, task_name, args.model, args.data_path, i, label, pred)

        all_labels, all_preds, skipped = build_metrics_from_records(records_by_index)

        # 每 20 条或最后一条打印进度
        if (i + 1) % 20 == 0 or (i + 1) == len(examples):
            done = len(all_labels)
            if done > 0:
                correct = sum(p == l for p, l in zip(all_preds, all_labels))
                logger.info(f"[{i+1}/{len(examples)}] 已完成 {done} 条，当前准确率: {correct/done:.4f}，跳过: {skipped}")
            else:
                logger.info(f"[{i+1}/{len(examples)}] 已完成 0 条，当前准确率: N/A，跳过: {skipped}")

        # 避免触发速率限制
        if args.request_delay > 0:
            time.sleep(args.request_delay)

    all_labels, all_preds, skipped = build_metrics_from_records(records_by_index)

    if not all_labels:
        logger.error("所有样本均无法解析，无法计算指标。")
        sys.exit(1)

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    logger.info("=" * 60)
    logger.info(f"评测结果 — {task_name} / {args.model}")
    logger.info(f"  样本总数:       {len(examples)}")
    logger.info(f"  有效预测:       {len(all_labels)}")
    logger.info(f"  跳过(无法解析): {skipped}")
    logger.info(f"  Accuracy:       {acc:.4f}")
    logger.info(f"  Macro Precision:{macro_precision:.4f}")
    logger.info(f"  Macro Recall:   {macro_recall:.4f}")
    logger.info(f"  Macro F1:       {macro_f1:.4f}")
    logger.info("=" * 60)

    # 可选：保存结果
    if out_dir is not None:
        result = {
            "task": task_name,
            "model": args.model,
            "timestamp": run_timestamp,
            "data_path": args.data_path,
            "total_samples": len(examples),
            "processed_samples": len(records_by_index),
            "valid_predictions": len(all_labels),
            "skipped": skipped,
            "resume_source": resume_source,
            "progress_path": str(progress_path) if progress_path is not None else None,
            "accuracy": acc,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }
        result_path = out_dir / f"{run_timestamp}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存至: {result_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Pure LLM evaluation for EvoShield")
    parser.add_argument("--task_name", choices=["PI", "JC", "SG"], default="JC",
                        help="任务名称: PI / JC / SG")
    parser.add_argument("--data_path", default="/home/han/llh/EvoShield/data/jailbreak-classification-balanced/full_test.csv",
                        help="测试集 CSV 路径")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="LLM 模型名称 (default: gpt-4o-mini)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最多评测样本数，不设则使用全部")
    parser.add_argument("--output_path", default="/home/han/llh/EvoShield/llm_log",
                        help="结果输出目录，日志和 JSON 都会写入该目录")
    parser.add_argument("--resume_from", default=None,
                        help="断点恢复来源，可传输出目录、.log 文件或 progress.jsonl")
    parser.add_argument("--request_delay", type=float, default=0.0,
                        help="每次请求后的等待秒数，用于控制速率 (default: 0)")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
