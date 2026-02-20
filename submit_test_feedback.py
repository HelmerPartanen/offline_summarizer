import requests
import json

url = "http://localhost:8000/feedback"
headers = {"Content-Type": "application/json"}

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

for fb in feedbacks:
    response = requests.post(url, headers=headers, data=json.dumps(fb))
    print(f"Submitted feedback for: {fb['text'][:50]}... Status: {response.status_code}, Response: {response.text}")
