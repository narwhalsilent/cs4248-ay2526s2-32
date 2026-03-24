import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def default_output_csv(model_path):
    model_name = os.path.basename(os.path.abspath(model_path.rstrip(os.sep)))
    return os.path.join("outputs", "evaluation", model_name, "test_predictions.csv")


def calculate_perplexity(texts, model, tokenizer):
    """Calculates GPT-2 perplexity for fluency."""
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt", truncation=True)
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


def build_inputs(factual_texts, source_prefix):
    if source_prefix:
        return [source_prefix + text for text in factual_texts]
    return factual_texts


def generate_predictions(model, tokenizer, factual_texts, source_prefix, device, batch_size, max_length, num_beams):
    model_inputs = build_inputs(factual_texts, source_prefix)
    generated_texts = []

    for start_idx in tqdm(range(0, len(model_inputs), batch_size), desc="Generating"):
        batch = model_inputs[start_idx:start_idx + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_length=max_length,
                num_beams=num_beams
            )

        generated_texts.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    return [text.strip() for text in generated_texts]


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = args.model_path or "checkpoints/bart_satire/final"

    print(f"Using device     : {device}")
    print(f"Checkpoint path  : {model_path}")
    print(f"Test data path   : {args.test_data}")

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {model_path}")
    if not os.path.exists(args.test_data):
        raise FileNotFoundError(f"Test set not found: {args.test_data}")

    test_df = pd.read_csv(args.test_data)
    required_columns = {"factual", "satirical"}
    missing_columns = required_columns - set(test_df.columns)
    if missing_columns:
        raise ValueError(f"Test CSV is missing required columns: {sorted(missing_columns)}")

    factual_texts = test_df["factual"].astype(str).tolist()
    reference_texts = test_df["satirical"].astype(str).tolist()

    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    # T5-style checkpoints need the task prefix used during training.
    source_prefix = args.source_prefix
    if source_prefix is None and "t5" in os.path.basename(model_path).lower():
        source_prefix = "translate factual to satire: "
    source_prefix = source_prefix or ""

    generated_texts = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        factual_texts=factual_texts,
        source_prefix=source_prefix,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_beams=args.num_beams,
    )

    print("Loading evaluators...")
    sarcasm_clf = pipeline(
        "text-classification",
        model="helinivan/english-sarcasm-detector",
        device=0 if device == "cuda" else -1
    )
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    gpt2_model.eval()

    print("Computing metrics...")
    sarcasm_results = sarcasm_clf(generated_texts, batch_size=args.batch_size)
    sarcasm_scores = [
        result["score"] if result["label"].lower() == "sarcastic" else 1 - result["score"]
        for result in sarcasm_results
    ]
    avg_sarcasm = float(np.mean(sarcasm_scores))

    embeddings_factual = sbert_model.encode(factual_texts, convert_to_tensor=True, show_progress_bar=True)
    embeddings_generated = sbert_model.encode(generated_texts, convert_to_tensor=True, show_progress_bar=True)
    cosine_scores = util.cos_sim(embeddings_factual, embeddings_generated)
    avg_cosine = torch.mean(torch.diag(cosine_scores)).item()

    ppl = calculate_perplexity(generated_texts, gpt2_model, gpt2_tokenizer)

    results_df = pd.DataFrame(
        {
            "factual": factual_texts,
            "reference_satirical": reference_texts,
            "generated_satirical": generated_texts,
            "sarcasm_score": sarcasm_scores,
        }
    )
    if not args.no_save_csv:
        output_csv = args.output_csv or default_output_csv(model_path)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        print(f"Saved detailed predictions to: {output_csv}")

    print("\n--- Evaluation Results ---")
    print(f"Num test examples                       : {len(test_df)}")
    print(f"Style Accuracy (RoBERTa sarcasm prob)   : {avg_sarcasm:.4f}")
    print(f"Content Preservation (SBERT cosine)     : {avg_cosine:.4f}")
    print(f"Linguistic Fluency (GPT-2 perplexity)   : {ppl:.2f}")

    sample_size = min(args.num_examples, len(test_df))
    if sample_size > 0:
        print(f"\n--- Random Sample Predictions ({sample_size}) ---")
        for row_idx in random.sample(range(len(test_df)), sample_size):
            print(f"\nExample {row_idx}")
            print(f"Factual   : {factual_texts[row_idx]}")
            print(f"Reference : {reference_texts[row_idx]}")
            print(f"Generated : {generated_texts[row_idx]}")
            print(f"Sarcasm   : {sarcasm_scores[row_idx]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned seq2seq satire model on the held-out test set.")
    parser.add_argument("--model_path", type=str, default="checkpoints/bart_satire/final", help="Path to a fine-tuned checkpoint directory.")
    parser.add_argument("--test_data", type=str, default="data/silver/test.csv", help="Path to held-out test CSV with factual and satirical columns.")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional path to save generated predictions and per-example scores.")
    parser.add_argument("--no_save_csv", action="store_true", help="Skip writing per-example predictions to disk.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation and classifier inference.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum generation length.")
    parser.add_argument("--num_beams", type=int, default=4, help="Beam width for decoding.")
    parser.add_argument("--num_examples", type=int, default=5, help="How many random example predictions to print.")
    parser.add_argument("--source_prefix", type=str, default=None, help="Optional task prefix. Leave unset for BART; set for T5 if needed.")
    args = parser.parse_args()
    main(args)
