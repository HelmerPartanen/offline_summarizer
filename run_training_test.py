import sys
import os

# Add relevant paths
sys.path.append(os.path.dirname(__file__))
from backend.summarizer import summarizer_instance

feedbacks = [
    {
        "text": "Artificial Intelligence is transforming the world. Large language models like GPT-4 are capable of reasoning, coding, and creative writing. However, they also pose risks such as bias and misinformation. We must develop safety guardrails to ensure they align with human values.",
        "rejected_summary": "AI is changing everything but it is also dangerous and bad if not controlled correctly.",
        "chosen_summary": "AI models like GPT-4 offer immense potential for reasoning and creativity but require safety guardrails to mitigate risks of bias and misinformation."
    },
    {
        "text": "The James Webb Space Telescope has captured stunning images of the early universe. By observing infrared light, it can see through dust clouds to reveal the birth of stars and galaxies. This is a major milestone in human understanding of the cosmos.",
        "rejected_summary": "The telescope saw stars through dust which is a cool thing for humans.",
        "chosen_summary": "The James Webb Space Telescope uses infrared light to observe the early universe and star formation, marking a significant milestone in astronomy."
    }
]

print("Logging test feedback...")
for fb in feedbacks:
    summarizer_instance.log_feedback(fb["text"], fb["rejected_summary"], fb["chosen_summary"])

print("Feedback logging complete.")

# Now run the training pipeline
print("\n--- Running Training Pipeline ---")
try:
    from training.prepare_feedback import prepare_data
    prepare_data()
    
    from training.train_dpo import train_dpo
    train_dpo()
except Exception as e:
    print(f"Error during training pipeline: {e}")
