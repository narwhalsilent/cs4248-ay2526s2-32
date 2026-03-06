import pandas as pd
import torch
import numpy as np
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import argparse
from tqdm import tqdm

def calculate_perplexity(texts, model, tokenizer):
    """Calculates GPT-2 perplexity for fluency."""
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return torch.exp(torch.stack(nlls).mean()).item()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Test Data (Source and Generated Outputs)
    # test_df = pd.read_csv("data/silver/test_predictions.csv")
    factual_texts = ["Local man forgets to buy milk."]
    generated_texts = ["Local man bravely embarks on perilous grocery quest, returns empty-handed."]
    
    # 2. Load Evaluators
    print("Loading Evaluators...")
    sarcasm_clf = pipeline("text-classification", model="helinivan/english-sarcasm-detector", device=0 if device=="cuda" else -1)
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    
    # 3. Evaluate Style (Sarcasm Detector)
    sarcasm_results = sarcasm_clf(generated_texts)
    sarcasm_scores = [res['score'] if res['label'] == 'sarcastic' else 1 - res['score'] for res in sarcasm_results]
    avg_sarcasm = np.mean(sarcasm_scores)
    
    # 4. Evaluate Content Preservation (SBERT)
    embeddings_factual = sbert_model.encode(factual_texts, convert_to_tensor=True)
    embeddings_generated = sbert_model.encode(generated_texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings_factual, embeddings_generated)
    avg_cosine = torch.mean(torch.diag(cosine_scores)).item()
    
    # 5. Evaluate Fluency (Perplexity)
    ppl = calculate_perplexity(generated_texts, gpt2_model, gpt2_tokenizer)
    
    print("\n--- Evaluation Results ---")
    print(f"Style Accuracy (RoBERTa Sarcasm Prob): {avg_sarcasm:.4f}")
    print(f"Content Preservation (SBERT Cosine):   {avg_cosine:.4f}")
    print(f"Linguistic Fluency (GPT-2 Perplexity): {ppl:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=False, help="Path to fine-tuned model")
    args = parser.parse_args()
    main(args)