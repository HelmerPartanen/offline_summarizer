import sys
import os

# Ensure we can import from the backend directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from summarizer import Summarizer

def main():
    print("--- Offline Summarizer Integration Example ---")
    
    # Example 1: Using the default model path
    print("\nInitializing summarizer with default model...")
    summarizer = Summarizer()
    
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. 
    Leading AI textbooks define the field as the study of "intelligent agents": any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
    Some popular accounts use the term "artificial intelligence" to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".
    """
    
    print("\nSummarizing sample text...")
    summary = summarizer.summarize(sample_text)
    print(f"\nSummary:\n{summary}")
    
    # Example 2: Loading a specific model (e.g., if you moved the trained_model folder)
    # custom_path = "/path/to/your/custom/model/folder"
    # if os.path.exists(custom_path):
    #     custom_summarizer = Summarizer(model_path=custom_path)
    #     print(custom_summarizer.summarize(sample_text))

if __name__ == "__main__":
    main()
