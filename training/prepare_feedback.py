import json
import os
import sys
import hashlib


MAX_PROMPT_CHARS = 1200


def _normalize_prompt(prompt):
    prompt = prompt.strip()
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS]
    if not prompt.startswith("summarize: "):
        return "summarize: " + prompt
    return prompt


def _sample_id(prompt, chosen, rejected):
    key = f"{prompt}\n{chosen}\n{rejected}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def _load_manifest(manifest_file):
    if not os.path.exists(manifest_file):
        return {
            "trained_ids": [],
            "last_run": None,
            "stats": {
                "total_feedback_rows": 0,
                "usable_pairs": 0,
                "unique_pairs": 0,
                "new_pairs": 0
            }
        }
    with open(manifest_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(manifest_file, manifest):
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def prepare_data():
    script_dir = os.path.dirname(__file__)
    feedback_file = os.path.join(script_dir, "..", "data", "feedback", "feedback.jsonl")
    output_file = os.path.join(script_dir, "..", "data", "feedback", "dpo_dataset.json")
    incremental_output_file = os.path.join(script_dir, "..", "data", "feedback", "dpo_dataset_incremental.json")
    manifest_file = os.path.join(script_dir, "..", "data", "feedback", "training_manifest.json")
    
    if not os.path.exists(feedback_file):
        print(f"No feedback file found at {feedback_file}")
        return {
            "total_feedback_rows": 0,
            "usable_pairs": 0,
            "unique_pairs": 0,
            "new_pairs": 0,
            "output_file": output_file,
            "incremental_output_file": incremental_output_file,
            "manifest_file": manifest_file
        }

    manifest = _load_manifest(manifest_file)
    trained_ids = set(manifest.get("trained_ids", []))

    total_feedback_rows = 0
    usable_pairs = 0
    dedup = {}

    with open(feedback_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_feedback_rows += 1
            data = json.loads(line)
            prompt = _normalize_prompt(data["prompt"])
            
            if data["chosen"] == data["rejected"]:
                 continue
            usable_pairs += 1

            sample = {
                "prompt": prompt,
                "chosen": data["chosen"],
                "rejected": data["rejected"]
            }
            sid = _sample_id(sample["prompt"], sample["chosen"], sample["rejected"])
            sample["id"] = sid
            dedup[sid] = sample

    full_dataset = list(dedup.values())
    new_dataset = [sample for sample in full_dataset if sample["id"] not in trained_ids]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_dataset, f, ensure_ascii=False, indent=2)
    with open(incremental_output_file, "w", encoding="utf-8") as f:
        json.dump(new_dataset, f, ensure_ascii=False, indent=2)

    manifest["stats"] = {
        "total_feedback_rows": total_feedback_rows,
        "usable_pairs": usable_pairs,
        "unique_pairs": len(full_dataset),
        "new_pairs": len(new_dataset)
    }
    _save_manifest(manifest_file, manifest)

    print(
        f"Prepared full dataset: {len(full_dataset)} pairs -> {output_file}\n"
        f"Prepared incremental dataset: {len(new_dataset)} new pairs -> {incremental_output_file}"
    )

    return {
        "total_feedback_rows": total_feedback_rows,
        "usable_pairs": usable_pairs,
        "unique_pairs": len(full_dataset),
        "new_pairs": len(new_dataset),
        "output_file": output_file,
        "incremental_output_file": incremental_output_file,
        "manifest_file": manifest_file
    }

if __name__ == "__main__":
    prepare_data()
