from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import sys
import os
import json
import argparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# Ensure we can import config from the same directory
sys.path.append(os.path.dirname(__file__))
import config


def _resolve_base_model(script_dir):
    model_dir = os.path.join(script_dir, "..", "models")
    pointer_files = [
        os.path.join(model_dir, "active_model.txt"),
        os.path.join(model_dir, "active_dpo_model.txt"),
    ]

    for pointer_file in pointer_files:
        if os.path.exists(pointer_file):
            with open(pointer_file, "r", encoding="utf-8") as f:
                pointed = f.read().strip()
            if pointed and os.path.exists(os.path.join(pointed, "config.json")):
                return pointed

    candidates = [
        os.path.join(model_dir, "trained_model_new"),
        os.path.join(model_dir, "trained_model"),
        os.path.join(model_dir, "dpo_model"),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "config.json")):
            return candidate

    return config.MODEL_NAME


def _mark_active_model(script_dir, model_path):
    pointer = os.path.join(script_dir, "..", "models", "active_model.txt")
    with open(pointer, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(model_path))


def _reload_backend_if_running():
    payload = json.dumps({}).encode("utf-8")
    request = Request(
        "http://localhost:8000/reload_model",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            return {"ok": True, "response": response.read().decode("utf-8")}
    except HTTPError as exc:
        return {"ok": False, "reason": f"HTTP {exc.code}: {exc.reason}"}
    except URLError as exc:
        return {"ok": False, "reason": f"Backend not reachable: {exc.reason}"}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def train(max_articles=None, epochs=None, no_reload=False):
    # Load dataset
    # Expecting data in ../data/processed relative to this script
    script_dir = os.path.dirname(__file__)
    processed_dir = os.path.join(script_dir, "..", "data", "processed")
    data_file = os.path.join(processed_dir, "cnn_dailymail_train.json")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"{data_file} not found. Run prepare_data.py first.")

    dataset = load_dataset("json", data_files={"train": data_file})

    train_dataset = dataset["train"]
    if max_articles is not None and max_articles > 0:
        subset_size = min(max_articles, len(train_dataset))
        train_dataset = train_dataset.select(range(subset_size))
        print(f"Using {subset_size} CNN/DailyMail articles for this training run.")
    else:
        print(f"Using full CNN/DailyMail dataset with {len(train_dataset)} articles.")

    model_name_or_path = _resolve_base_model(script_dir)
    print(f"Loading base model from {model_name_or_path}")

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

    # Preprocessing
    def preprocess(batch):
        prefixed_text = [f"summarize: {text}" for text in batch["text"]]
        inputs = tokenizer(prefixed_text, truncation=True, max_length=config.MAX_INPUT_LENGTH)
        targets = tokenizer(batch["summary"], truncation=True, max_length=config.MAX_SUMMARY_LENGTH)
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized_dataset = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    # Training arguments
    run_stamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(script_dir, "..", "models", "cnn_model_runs", f"run_{run_stamp}")
    num_epochs = epochs if epochs is not None else config.EPOCHS

    remaining_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.BATCH_SIZE,
        num_train_epochs=num_epochs,
        save_steps=50,
        save_total_limit=2,
        logging_steps=20,
        learning_rate=config.LEARNING_RATE,
        report_to=[],
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=remaining_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Start training
    trainer.train()
    
    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    _mark_active_model(script_dir, output_dir)
    print(f"Saved CNN-trained model to {output_dir}")

    if not no_reload:
        reload_result = _reload_backend_if_running()
        if reload_result.get("ok"):
            print("Backend model reload succeeded.")
        else:
            print(f"Backend reload skipped/failed: {reload_result.get('reason')}")

    return {
        "trained": True,
        "articles_used": len(train_dataset),
        "epochs": num_epochs,
        "output_model_path": output_dir,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train summarizer on CNN/DailyMail with configurable article count.")
    parser.add_argument("--max-articles", type=int, default=None, help="Train on first N articles only.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--no-reload", action="store_true", help="Skip backend model reload after training.")
    args = parser.parse_args()

    result = train(max_articles=args.max_articles, epochs=args.epochs, no_reload=args.no_reload)
    print(json.dumps(result, indent=2))
