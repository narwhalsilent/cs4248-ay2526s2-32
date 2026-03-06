import os
import json
import pandas as pd
import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from evaluate import load
# Assuming evaluate.py contains your tri-factor metric functions
# from evaluate import calculate_perplexity, evaluate_style, evaluate_content

def run_ablation_sweep(test_data_path, models_dir):
    """
    Evaluates multiple trained models (ablations) on the same test set
    to compare model sizes, loss functions, or input formats.
    """
    # Load test data
    df = pd.read_csv(test_data_path)
    source_texts = df['factual'].tolist()
    
    results = []
    
    # Iterate through all fine-tuned models in the checkpoints directory
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if not os.path.isdir(model_path):
            continue
            
        print(f"\nEvaluating ablation model: {model_name}")
        
        # Load specific ablation model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda")
        
        # Generate predictions
        generated_texts = []
        for text in source_texts:
            inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True).to("cuda")
            outputs = model.generate(**inputs, max_length=128, num_beams=4)
            generated_texts.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
            
        # Calculate Metrics (Mocked calls to your evaluation framework)
        # style_score = evaluate_style(generated_texts)
        # content_score = evaluate_content(source_texts, generated_texts)
        # fluency_score = calculate_perplexity(generated_texts, gpt2_model, gpt2_tokenizer)
        
        # Mock metrics for demonstration
        style_score, content_score, fluency_score = 0.85, 0.75, 25.4 
        
        results.append({
            "model_config": model_name,
            "style_accuracy": style_score,
            "content_preservation": content_score,
            "perplexity": fluency_score
        })
        
    # Save ablation results
    results_df = pd.DataFrame(results)
    results_df.to_csv("data/silver/ablation_results.csv", index=False)
    print("\nAblation summary saved to data/silver/ablation_results.csv")
    print(results_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="data/silver/test.csv")
    parser.add_argument("--models_dir", type=str, default="checkpoints/")
    args = parser.parse_args()
    
    run_ablation_sweep(args.test_data, args.models_dir)