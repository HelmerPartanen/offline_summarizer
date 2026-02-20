import os
import json
import argparse
from datetime import datetime

from datasets import load_dataset, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


def resolve_active_model(script_dir):
    model_dir = os.path.join(script_dir, "..", "models")
    pointers = [
        os.path.join(model_dir, "active_model.txt"),
        os.path.join(model_dir, "active_dpo_model.txt"),
    ]

    for pointer in pointers:
        if os.path.exists(pointer):
            with open(pointer, "r", encoding="utf-8") as f:
                path = f.read().strip()
            if path and os.path.exists(os.path.join(path, "config.json")):
                return path

    candidates = [
        os.path.join(model_dir, "cnn_model_runs"),
        os.path.join(model_dir, "sft_model_runs"),
        os.path.join(model_dir, "dpo_model_runs"),
        os.path.join(model_dir, "trained_model"),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "config.json")):
            return candidate

    return "t5-small"


def _load_records_fallback(data_file):
    with open(data_file, "r", encoding="utf-8") as f:
        raw = f.read()

    records = []
    chunks = [chunk.strip() for chunk in raw.split('{"id":') if chunk.strip()]
    for chunk in chunks:
        rebuilt = '{"id":' + chunk
        if not rebuilt.endswith("}"):
            continue
        try:
            records.append(json.loads(rebuilt))
        except Exception:
            continue
    return records


def load_eval_dataset(data_file):
    try:
        return load_dataset("json", data_files={"eval": data_file})["eval"]
    except Exception:
        records = _load_records_fallback(data_file)
        return Dataset.from_list(records)


def tokenize(text):
    return [token for token in text.lower().strip().split() if token]


def token_f1(pred, ref):
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    pred_counts = {}
    ref_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    overlap = 0
    for t, count in pred_counts.items():
        overlap += min(count, ref_counts.get(t, 0))

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def lcs_length(a_tokens, b_tokens):
    n = len(a_tokens)
    m = len(b_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a_tokens[i - 1] == b_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def rouge_l_f1(pred, ref):
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    p = lcs / len(pred_tokens)
    r = lcs / len(ref_tokens)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def evaluate_model(model_path, data_file, max_samples=20, max_input_length=384, max_summary_length=128):
    dataset = load_eval_dataset(data_file)
    if len(dataset) == 0:
        raise ValueError("Evaluation dataset is empty")

    subset_size = min(max_samples, len(dataset)) if max_samples and max_samples > 0 else len(dataset)
    dataset = dataset.select(range(subset_size))

    required_cols = {"source_text", "summary"}
    missing = [c for c in required_cols if c not in dataset.column_names]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_rouge_l = 0.0

    for row in dataset:
        category = row.get("category", "general")
        prompt = f"summarize ({category}): {row['source_text']}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(device)

        output_ids = model.generate(
            inputs["input_ids"],
            max_length=max_summary_length,
            min_length=20,
            num_beams=2,
            early_stopping=True,
        )
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        ref = row["summary"]

        p, r, f1 = token_f1(pred, ref)
        rl = rouge_l_f1(pred, ref)

        total_precision += p
        total_recall += r
        total_f1 += f1
        total_rouge_l += rl

    count = len(dataset)
    return {
        "samples": count,
        "token_precision": round(total_precision / count, 4),
        "token_recall": round(total_recall / count, 4),
        "token_f1": round(total_f1 / count, 4),
        "rouge_l_f1": round(total_rouge_l / count, 4),
    }


def save_eval_result(script_dir, result):
    eval_dir = os.path.join(script_dir, "..", "data", "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    history_file = os.path.join(eval_dir, "eval_history.jsonl")
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return history_file


def run(data_file, model_path=None, max_samples=20):
    script_dir = os.path.dirname(__file__)
    resolved_model = model_path if model_path else resolve_active_model(script_dir)

    result = evaluate_model(
        model_path=resolved_model,
        data_file=data_file,
        max_samples=max_samples,
    )

    payload = {
        "timestamp": datetime.now().isoformat(),
        "model_path": os.path.abspath(resolved_model),
        "dataset": os.path.abspath(data_file),
        "metrics": result,
    }
    history_file = save_eval_result(script_dir, payload)
    payload["history_file"] = os.path.abspath(history_file)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate summarization model quality on a labeled dataset.")
    parser.add_argument(
        "--data-file",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "training_dataset",
            "website_summarization_training_data.jsonl",
        ),
        help="Path to labeled evaluation data file",
    )
    parser.add_argument("--model-path", default=None, help="Optional model path override")
    parser.add_argument("--max-samples", type=int, default=20, help="Max samples to evaluate")
    args = parser.parse_args()

    result = run(args.data_file, args.model_path, args.max_samples)
    print(json.dumps(result, indent=2))
