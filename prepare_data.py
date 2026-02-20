from datasets import load_dataset
import json
import os

def prepare_data():
    print("Loading CNN/DailyMail dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    print("Processing training data...")
    train_data = []
    
    # Process a subset or full set. 
    # User said "thousands of article-summary pairs". 
    # Processing all might take time, but let's stick to the script.
    # We'll limit it for now to ensure it runs quickly for demonstration if needed, 
    # but the user asked for the code to convert it.
    
    for item in dataset['train']:
        train_data.append({
            "text": item['article'],
            "summary": item['highlights']
        })

    output_dir = os.path.join(os.path.dirname(__file__), "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "cnn_dailymail_train.json")

    print(f"Saving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    
    print("Data preparation complete.")

if __name__ == "__main__":
    prepare_data()
