from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import os
import threading

# Define paths relative to this file
# This ensures it works regardless of where the script is executed from
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "trained_model")
NEW_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "trained_model_new")
DPO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "dpo_model")
ACTIVE_MODEL_FILE = os.path.join(os.path.dirname(__file__), "..", "models", "active_model.txt")
ACTIVE_DPO_MODEL_FILE = os.path.join(os.path.dirname(__file__), "..", "models", "active_dpo_model.txt")

def resolve_best_model_path():
    if os.path.exists(ACTIVE_MODEL_FILE):
        with open(ACTIVE_MODEL_FILE, "r", encoding="utf-8") as f:
            active_path = f.read().strip()
        if active_path and os.path.exists(os.path.join(active_path, "config.json")):
            print(f"Found active model at {active_path}")
            return active_path

    if os.path.exists(ACTIVE_DPO_MODEL_FILE):
        with open(ACTIVE_DPO_MODEL_FILE, "r", encoding="utf-8") as f:
            active_path = f.read().strip()
        if active_path and os.path.exists(os.path.join(active_path, "config.json")):
            print(f"Found active DPO model at {active_path}")
            return active_path

    if os.path.exists(os.path.join(DPO_MODEL_PATH, "config.json")):
        print(f"Found DPO-tuned model at {DPO_MODEL_PATH}")
        return DPO_MODEL_PATH
    if os.path.exists(os.path.join(NEW_MODEL_PATH, "config.json")):
        print(f"Found recently trained model at {NEW_MODEL_PATH}")
        return NEW_MODEL_PATH
    if os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        return MODEL_PATH
    return "t5-small"

class Summarizer:
    def __init__(self, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path if model_path else resolve_best_model_path()
        self._lock = threading.Lock()
        self.tokenizer = None
        self.model = None
        self._load_model(self.model_path)

    def _load_model(self, candidate_path):
        print(f"Loading model on {self.device}...")

        try:
            if os.path.exists(os.path.join(candidate_path, "config.json")):
                print(f"Found trained model at {candidate_path}")
                tokenizer = T5Tokenizer.from_pretrained(candidate_path)
                model = T5ForConditionalGeneration.from_pretrained(candidate_path).to(self.device)
                self.tokenizer = tokenizer
                self.model = model
                self.model_path = candidate_path
            else:
                print(f"Trained model not found or empty at {candidate_path}. Loading base model 't5-small'.")
                tokenizer = T5Tokenizer.from_pretrained("t5-small")
                model = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)
                self.tokenizer = tokenizer
                self.model = model
                self.model_path = "t5-small"
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def reload_model(self, model_path=None):
        with self._lock:
            target_path = model_path if model_path else resolve_best_model_path()
            self._load_model(target_path)
            return {
                "status": "reloaded",
                "model_path": self.model_path,
                "device": self.device
            }

    def summarize(self, text, max_length=150):
        # T5 uses "summarize: " prefix for this task usually.
        # Adding it improves performance even on fine-tuned models if base was T5.
        input_text = "summarize: " + text if not text.startswith("summarize:") else text
        
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=30,
            num_beams=4,
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def log_feedback(self, text, rejected_summary, chosen_summary):
        import json
        from datetime import datetime
        
        feedback_dir = os.path.join(os.path.dirname(__file__), "..", "data", "feedback")
        os.makedirs(feedback_dir, exist_ok=True)
        feedback_file = os.path.join(feedback_dir, "feedback.jsonl")
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": text,
            "rejected": rejected_summary,
            "chosen": chosen_summary
        }
        
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"--- FEEDBACK LOGGED ---")
        print(f"File: {os.path.abspath(feedback_file)}")
        print(f"Entry: {json.dumps(entry, indent=2)}")
        print(f"-----------------------")


# Initialize with default path for the app
summarizer_instance = Summarizer()

def summarize(text, max_length=150):
    return summarizer_instance.summarize(text, max_length)


def reload_model(model_path=None):
    return summarizer_instance.reload_model(model_path)
