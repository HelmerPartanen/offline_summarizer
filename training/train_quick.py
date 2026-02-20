from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import sys
import os

# Ensure we can import config from the same directory
sys.path.append(os.path.dirname(__file__))
import config

def train():
    print("--- Quick Training Mode ---")
    # Load dataset
    script_dir = os.path.dirname(__file__)
    processed_dir = os.path.join(script_dir, "..", "data", "processed")
    data_file = os.path.join(processed_dir, "cnn_dailymail_train.json")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"{data_file} not found. Run prepare_data.py first.")

    dataset = load_dataset("json", data_files={"train": data_file})

    # Define model output directory
    output_dir = os.path.join(script_dir, "..", "models", "trained_model")

    # Check if we have an existing model to resume from
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"Resuming training from existing model at {output_dir}")
        model_name_or_path = output_dir
    else:
        print(f"Starting fresh training from base model {config.MODEL_NAME}")
        model_name_or_path = config.MODEL_NAME

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

    # Preprocessing
    def preprocess(batch):
        inputs = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=config.MAX_INPUT_LENGTH)
        targets = tokenizer(batch["summary"], truncation=True, padding="max_length", max_length=config.MAX_SUMMARY_LENGTH)
        inputs["labels"] = targets["input_ids"]
        return inputs

    # SUBSET: Take 100 random examples for quick testing
    print("Selecting 100 random examples for quick training...")
    # Removing seed=42 makes it random every time
    small_dataset = dataset["train"].shuffle().select(range(100))
    tokenized_dataset = small_dataset.map(preprocess, batched=True)

    # Training arguments
    output_dir = os.path.join(script_dir, "..", "models", "trained_model")

    # Windows-specific fix: prevent overwriting the model file that is currently loaded (memory-mapped)
    # This prevents [WinError 1224] The requested operation cannot be performed on a file with a user-mapped section open.
    if os.name == 'nt' and os.path.abspath(model_name_or_path) == os.path.abspath(output_dir):
        print(f"Windows detected: Changing output directory to avoid file locking conflict.")
        output_dir = output_dir + "_new"
        print(f"New output directory: {output_dir}")

    remaining_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.BATCH_SIZE,
        num_train_epochs=1, # Only 1 epoch for quick test
        save_steps=10,
        save_total_limit=1,
        logging_steps=5,
        learning_rate=config.LEARNING_RATE
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=remaining_args,
        train_dataset=tokenized_dataset
    )

    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Quick training complete!")

if __name__ == "__main__":
    train()
