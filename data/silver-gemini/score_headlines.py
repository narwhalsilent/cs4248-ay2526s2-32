import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bert_score import scorer
from tqdm import tqdm

def process_in_batches(data_list, batch_size):
    """Yields batches of a specified size from a list."""
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]

def score_candidates_gpu(data, device, tokenizer, model, bert_scorer, is_satirizing=True, batch_size=128):
    """Score candidates using flattened batched processing for GPU optimization."""

    ref_key = "factual_headline" if is_satirizing else "satirical_headline"

    # 1. Flatten the data structure for batch processing
    flat_candidates = []
    flat_references = []
    metadata = [] 

    for item_idx, item in enumerate(data):
        if ref_key not in item:
            print(f"Skipping item {item_idx}: Missing key '{ref_key}'")
            continue

        for i in range(1, 6):
            candidate_key = f"candidate_{i}"
            if candidate_key in item:
                flat_candidates.append(item[candidate_key])
                flat_references.append(item[ref_key])
                metadata.append({"item_idx": item_idx, "key": candidate_key, "text": item[candidate_key]})

    print(f"Flattened to {len(flat_candidates)} total candidates for processing.")

    sarcasm_scores_all = []

    # 2. Sarcasm Detection (Batched)
    print("Running Sarcasm Detection...")
    model.eval()
    with torch.no_grad():
        for batch_texts in tqdm(list(process_in_batches(flat_candidates, batch_size))):
            # Dynamic padding to longest in the specific batch
            inputs = tokenizer(
                batch_texts,
                padding=True, 
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)

            # Get probabilities and move back to CPU
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sarcasm_probs = probs[:, 1].cpu().tolist()
            sarcasm_scores_all.extend(sarcasm_probs)

    # 3. BERTScore (Batched)
    print("Running BERTScore...")
    _, _, bert_f1 = bert_scorer.score(
        flat_candidates,
        flat_references,
        verbose=True,
        batch_size=batch_size
    )
    bert_f1_scores = bert_f1.cpu().tolist()

    # 4. Reconstruct the original JSON structure
    print("Reconstructing JSON payload...")
    results = []
    for item in data:
        results.append({
            "original_headline": item.get(ref_key, ""),
            "candidates": []
        })

    for meta, sarcasm_score, f1_score in zip(metadata, sarcasm_scores_all, bert_f1_scores):
        idx = meta["item_idx"]
        style_score = sarcasm_score if is_satirizing else (1.0 - sarcasm_score)

        results[idx]["candidates"].append({
            "candidate": meta["key"],
            "text": meta["text"],
            "content_score": f1_score,
            "style_score": style_score
        })

    return results

if __name__ == "__main__":

    INPUT_DIR = "../data/silver-gemini" 

    OUTPUT_DIR = "../data/silver-gemini"
    BATCH_SIZE = 128

    # --- GPU Setup ---
    print("Initializing Device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cpu":
        print("WARNING: No GPU detected! Please turn on the GPU in your Kaggle Notebook settings.")

    # --- Load Models to GPU ---
    print("Loading Sarcasm Model to GPU (float16 precision)...")
    model_name = "helinivan/english-sarcasm-detector"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to(device)

    print("Loading BERTScorer...")
    bert_scorer = scorer.BERTScorer(
        lang="en",
        rescale_with_baseline=True,
        device=device,
    )

    # Process satirizing headlines
    satirizing_file = os.path.join(INPUT_DIR, "prettified_satirized_headlines.json")
    if os.path.exists(satirizing_file):
        print(f"\n--- Processing Satirizing Data ---")
        with open(satirizing_file, 'r') as f:
            satirizing_data = json.load(f)
        satirizing_scores = score_candidates_gpu(
            satirizing_data, device, tokenizer, model, bert_scorer,
            is_satirizing=True, batch_size=BATCH_SIZE
        )
        output_path = os.path.join(OUTPUT_DIR, "scored_satirized_headlines.json")
        with open(output_path, 'w') as f:
            json.dump(satirizing_scores, f, indent=2)
        print(f"Saved satirizing scores to {output_path}")

    # Process desatirizing headlines
    desatirizing_file = os.path.join(INPUT_DIR, "prettified_desatirized_headlines.json")
    if os.path.exists(desatirizing_file):
        print(f"\n--- Processing Desatirizing Data ---")
        with open(desatirizing_file, 'r') as f:
            desatirizing_data = json.load(f)
        desatirizing_scores = score_candidates_gpu(
            desatirizing_data, device, tokenizer, model, bert_scorer,
            is_satirizing=False, batch_size=BATCH_SIZE
        )
        output_path = os.path.join(OUTPUT_DIR, "scored_desatirized_headlines.json")
        with open(output_path, 'w') as f:
            json.dump(desatirizing_scores, f, indent=2)
        print(f"Saved desatirizing scores to {output_path}")