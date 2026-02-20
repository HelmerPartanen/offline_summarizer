import os
import sys
import json
import argparse
import re
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from datasets import load_dataset, Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_utils import get_last_checkpoint

# Ensure local imports work
sys.path.append(os.path.dirname(__file__))
import config


def trigger_backend_reload(reload_url="http://localhost:8000/reload_model"):
    payload = json.dumps({}).encode("utf-8")
    request = Request(
        reload_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            body = response.read().decode("utf-8")
            return {"ok": True, "response": body}
    except HTTPError as exc:
        return {"ok": False, "reason": f"HTTP {exc.code}: {exc.reason}"}
    except URLError as exc:
        return {"ok": False, "reason": f"Backend not reachable: {exc.reason}"}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def resolve_base_model(script_dir):
    model_dir = os.path.join(script_dir, "..", "models")
    active_model_file = os.path.join(model_dir, "active_model.txt")
    active_dpo_file = os.path.join(model_dir, "active_dpo_model.txt")

    for pointer_file in [active_model_file, active_dpo_file]:
        if os.path.exists(pointer_file):
            with open(pointer_file, "r", encoding="utf-8") as f:
                pointed = f.read().strip()
            if pointed and os.path.exists(os.path.join(pointed, "config.json")):
                return pointed

    candidates = [
        os.path.join(model_dir, "dpo_model"),
        os.path.join(model_dir, "trained_model_new"),
        os.path.join(model_dir, "trained_model"),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "config.json")):
            return candidate

    return config.MODEL_NAME


def _load_records_fallback(data_file):
    with open(data_file, "r", encoding="utf-8") as f:
        raw = f.read()

    records = []
    chunks = [chunk.strip() for chunk in re.split(r'(?=\{\s*"id"\s*:)', raw) if chunk.strip()]

    for chunk in chunks:
        id_match = re.search(r'"id"\s*:\s*(\d+)', chunk)
        category_match = re.search(r'"category"\s*:\s*"([^"]+)"', chunk)

        source_key = '"source_text": "'
        summary_key = '", "summary": "'

        source_start = chunk.find(source_key)
        summary_split = chunk.rfind(summary_key)
        summary_end = chunk.rfind('"}')

        if (
            not id_match
            or not category_match
            or source_start == -1
            or summary_split == -1
            or summary_end == -1
            or summary_split <= source_start
        ):
            continue

        source_start += len(source_key)
        source_text = chunk[source_start:summary_split]
        summary = chunk[summary_split + len(summary_key):summary_end]

        records.append(
            {
                "id": int(id_match.group(1)),
                "category": category_match.group(1),
                "source_text": source_text,
                "summary": summary,
            }
        )

    if not records:
        raise ValueError("No JSON objects parsed from training data file.")
    return records


def load_training_dataset(data_file):
    try:
        return load_dataset("json", data_files={"train": data_file})["train"]
    except Exception as first_error:
        print(f"Strict JSONL loading failed, using robust parser fallback: {first_error}")
        records = _load_records_fallback(data_file)
        return Dataset.from_list(records)


def mark_active_model(script_dir, model_path):
    active_model_file = os.path.join(script_dir, "..", "models", "active_model.txt")
    with open(active_model_file, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(model_path))


def train_from_jsonl(
    data_file,
    epochs=3,
    batch_size=1,
    learning_rate=5e-5,
    max_input_length=384,
    max_summary_length=128,
    save_steps=5,
    logging_steps=5,
    max_steps=-1,
    reload_backend=True,
):
    script_dir = os.path.dirname(__file__)

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Training data not found: {data_file}")

    print("=== Automatic JSONL Supervised Training ===")
    print(f"Data: {data_file}")

    dataset = load_training_dataset(data_file)
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty.")

    required = {"source_text", "summary"}
    missing = [name for name in required if name not in dataset.column_names]
    if missing:
        raise ValueError(f"Missing required fields in dataset: {missing}")

    base_model_path = resolve_base_model(script_dir)
    print(f"Base model: {base_model_path}")

    tokenizer = T5Tokenizer.from_pretrained(base_model_path)
    model = T5ForConditionalGeneration.from_pretrained(base_model_path)

    def preprocess(batch):
        categories = batch.get("category", ["general"] * len(batch["source_text"]))
        prompts = [
            f"summarize ({category}): {text.strip()}"
            for category, text in zip(categories, batch["source_text"])
        ]
        model_inputs = tokenizer(
            prompts,
            truncation=True,
            max_length=max_input_length,
        )

        labels = tokenizer(
            batch["summary"],
            truncation=True,
            max_length=max_summary_length,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing website training data",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    work_dir = os.path.join(script_dir, "..", "models", "sft_auto_work")
    os.makedirs(work_dir, exist_ok=True)

    latest_checkpoint = get_last_checkpoint(work_dir)
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
    else:
        print("No previous checkpoint found, starting fresh run.")

    training_args = TrainingArguments(
        output_dir=work_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=logging_steps,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        max_steps=max_steps,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    interrupted = False
    try:
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    except KeyboardInterrupt:
        interrupted = True
        print("Training interrupted by user. Saving current progress...")
    finally:
        run_stamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        final_dir = os.path.join(script_dir, "..", "models", "sft_model_runs", f"run_{run_stamp}")
        os.makedirs(final_dir, exist_ok=True)

        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        mark_active_model(script_dir, final_dir)

        print(f"Saved model to: {final_dir}")
        print("Active model pointer updated.")

        reload_result = None
        if reload_backend:
            reload_result = trigger_backend_reload()
            if reload_result.get("ok"):
                print("Backend model reload succeeded.")
            else:
                print(f"Backend reload skipped/failed: {reload_result.get('reason')}")

    return {
        "interrupted": interrupted,
        "base_model": base_model_path,
        "active_model": final_dir,
        "dataset_rows": len(dataset),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic supervised training from JSONL website-style data.")
    parser.add_argument(
        "--data-file",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "training_dataset",
            "website_summarization_training_data.jsonl",
        ),
        help="Path to JSONL file containing source_text and summary fields.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-input-length", type=int, default=384)
    parser.add_argument("--max-summary-length", type=int, default=128)
    parser.add_argument("--save-steps", type=int, default=5)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Set to a positive number for short test runs.",
    )
    parser.add_argument("--no-reload", action="store_true", help="Skip backend reload endpoint call.")

    args = parser.parse_args()

    result = train_from_jsonl(
        data_file=args.data_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_input_length=args.max_input_length,
        max_summary_length=args.max_summary_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps,
        reload_backend=not args.no_reload,
    )
    print(json.dumps(result, indent=2))
