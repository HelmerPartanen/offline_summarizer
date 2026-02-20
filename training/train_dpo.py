import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import sys
import json
from datetime import datetime

# Add script directory to path
sys.path.append(os.path.dirname(__file__))
import config


def _resolve_model_path(script_dir):
    active_model_file = os.path.join(script_dir, "..", "models", "active_model.txt")
    if os.path.exists(active_model_file):
        with open(active_model_file, "r", encoding="utf-8") as f:
            active_path = f.read().strip()
        if active_path and os.path.exists(os.path.join(active_path, "config.json")):
            return active_path

    active_model_file = os.path.join(script_dir, "..", "models", "active_dpo_model.txt")
    if os.path.exists(active_model_file):
        with open(active_model_file, "r", encoding="utf-8") as f:
            active_path = f.read().strip()
        if active_path and os.path.exists(os.path.join(active_path, "config.json")):
            return active_path

    dpo_model_path = os.path.join(script_dir, "..", "models", "dpo_model")
    new_model_path = os.path.join(script_dir, "..", "models", "trained_model_new")
    trained_model_path = os.path.join(script_dir, "..", "models", "trained_model")

    for candidate in [dpo_model_path, new_model_path, trained_model_path]:
        if os.path.exists(os.path.join(candidate, "config.json")):
            return candidate
    return config.MODEL_NAME


def _choose_epochs(dataset_size):
    if dataset_size <= 5:
        return 6
    if dataset_size <= 20:
        return 4
    return 2


def _choose_epochs_fast(dataset_size):
    if dataset_size <= 8:
        return 1
    return 1


def _build_output_dir(script_dir):
    runs_dir = os.path.join(script_dir, "..", "models", "dpo_model_runs")
    os.makedirs(runs_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(runs_dir, f"run_{stamp}")
    active_model_file = os.path.join(script_dir, "..", "models", "active_dpo_model.txt")
    generic_active_model_file = os.path.join(script_dir, "..", "models", "active_model.txt")
    return output_dir, active_model_file, generic_active_model_file


def _update_manifest_after_training(script_dir, trained_ids, base_model_path, output_dir, dataset_size):
    manifest_file = os.path.join(script_dir, "..", "data", "feedback", "training_manifest.json")
    manifest = {
        "trained_ids": [],
        "last_run": None,
        "stats": {}
    }
    if os.path.exists(manifest_file):
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    existing = set(manifest.get("trained_ids", []))
    existing.update(trained_ids)
    manifest["trained_ids"] = sorted(existing)
    manifest["last_run"] = {
        "timestamp": datetime.now().isoformat(),
        "mode": "incremental_dpo",
        "trained_new_pairs": dataset_size,
        "base_model_path": base_model_path,
        "output_model_path": output_dir
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def train_dpo(data_file=None, fast_mode=True, max_train_samples=None):
    print("--- DPO Training Mode ---")
    script_dir = os.path.dirname(__file__)
    if data_file is None:
        data_file = os.path.join(script_dir, "..", "data", "feedback", "dpo_dataset_incremental.json")
    
    if not os.path.exists(data_file):
        print(f"Dataset {data_file} not found. Run prepare_feedback.py first.")
        return {"trained": False, "reason": "dataset_not_found", "data_file": data_file}

    # Load dataset
    dataset = load_dataset("json", data_files=data_file, split="train")
    print(f"Found {len(dataset)} examples in the dataset.")
    
    if len(dataset) < 1:
        print("No new preference pairs to train. Model is already up to date with current feedback.")
        return {"trained": False, "reason": "no_new_pairs", "data_file": data_file}

    if max_train_samples is not None and max_train_samples > 0 and len(dataset) > max_train_samples:
        dataset = dataset.select(range(max_train_samples))
        print(f"Capped training set to {len(dataset)} newest pending pairs for this run.")

    # Load model and tokenizer
    model_path = _resolve_model_path(script_dir)
        
    print(f"Loading model from {model_path}...")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    # DPO requires a reference model (frozen) to compare against
    # If we use the same model, DPOTrainer will handle cloning/freezing
    ref_model = T5ForConditionalGeneration.from_pretrained(model_path)

    # Output directory for DPO-tuned model (versioned to avoid Windows file locking)
    output_dir, active_model_file, generic_active_model_file = _build_output_dir(script_dir)

    num_epochs = _choose_epochs_fast(len(dataset)) if fast_mode else _choose_epochs(len(dataset))
    print(f"Training for {num_epochs} epochs...")

    # In trl >= 0.12.0, beta, max_length etc. are in DPOConfig
    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=num_epochs,
        logging_steps=max(1, len(dataset)),
        save_steps=max(1000, len(dataset) * 10),
        learning_rate=2e-4 if fast_mode else 1e-4,
        remove_unused_columns=False, # Important for DPOTrainer
        gradient_accumulation_steps=1,
        beta=0.1,
        max_length=80 if fast_mode else config.MAX_SUMMARY_LENGTH,
        max_prompt_length=192 if fast_mode else config.MAX_INPUT_LENGTH,
        report_to=[],
        use_cpu=True  # Explicitly use CPU as GPU/bf16 is not supported on this setup
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args
    )

    print("Starting DPO training...")
    dpo_trainer.train()

    print(f"Saving DPO-tuned model to {output_dir}")
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(active_model_file, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(output_dir))
    with open(generic_active_model_file, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(output_dir))

    trained_ids = [row.get("id") for row in dataset if row.get("id")]
    _update_manifest_after_training(script_dir, trained_ids, model_path, output_dir, len(dataset))
    print("DPO training complete!")

    return {
        "trained": True,
        "trained_pairs": len(dataset),
        "base_model_path": model_path,
        "output_model_path": output_dir,
        "data_file": data_file,
        "fast_mode": fast_mode
    }

if __name__ == "__main__":
    train_dpo()
