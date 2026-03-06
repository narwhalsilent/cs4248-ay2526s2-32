import pandas as pd
import torch
from transformers import pipeline
from evaluate import load
import argparse

def generate_candidates(prompt, teacher_model, num_return_sequences=5):
    """
    Mock function to query Llama 3 for candidates using Pragmatic Metacognitive Prompting.
    Replace with actual API call or local Llama 3 generation pipeline.
    """
    # Example using a local pipeline:
    # responses = teacher_model(prompt, num_return_sequences=num_return_sequences, max_length=50)
    # return [res['generated_text'] for res in responses]
    return [f"Satirical version {i} of: {prompt}" for i in range(num_return_sequences)]

def main(args):
    # 1. Load factual headlines (HuffPost)
    # df = pd.read_csv('data/raw/factual_headlines.csv')
    factual_headlines = ["Local man forgets to buy milk.", "City council approves new budget."]
    
    # 2. Initialize Evaluators
    print("Loading Sarcasm Detector and BERTScore...")
    sarcasm_classifier = pipeline("text-classification", model="helinivan/english-sarcasm-detector", device=0 if torch.cuda.is_available() else -1)
    bertscore = load("bertscore")
    
    silver_dataset = []

    for headline in factual_headlines:
        # Generate N=5 candidates
        candidates = generate_candidates(headline, teacher_model=None, num_return_sequences=5)
        
        best_candidate = None
        best_score = -1
        
        for cand in candidates:
            # Score 1: Style Accuracy (Sarcasm probability)
            sarcasm_result = sarcasm_classifier(cand)[0]
            style_score = sarcasm_result['score'] if sarcasm_result['label'] == 'sarcastic' else 1 - sarcasm_result['score']
            
            # Score 2: Content Preservation (BERTScore F1)
            bs_result = bertscore.compute(predictions=[cand], references=[headline], lang="en")
            content_score = bs_result['f1'][0]
            
            # Harmonic mean or weighted average
            combined_score = (style_score + content_score) / 2
            
            if combined_score > best_score:
                best_score = combined_score
                best_candidate = cand
                
        if best_candidate:
            silver_dataset.append({"factual": headline, "satirical": best_candidate, "score": best_score})
            
    # Save the generated "Silver Dataset"
    silver_df = pd.DataFrame(silver_dataset)
    silver_df.to_csv("data/silver/silver_dataset.csv", index=False)
    print("Silver dataset generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    args = parser.parse_args()
    main(args)