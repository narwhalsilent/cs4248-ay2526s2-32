import argparse
import json
import os
import random
import re
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def default_output_csv(model_path):
    abs_path = os.path.abspath(model_path.rstrip(os.sep))
    model_name = os.path.basename(abs_path)

    if model_name == "final" or model_name.startswith("checkpoint-"):
        model_name = os.path.basename(os.path.dirname(abs_path))

    return os.path.join("outputs", "evaluation", model_name, "test_predictions.csv")


def default_preference_jsonl(model_path):
    abs_path = os.path.abspath(model_path.rstrip(os.sep))
    model_name = os.path.basename(abs_path)

    if model_name == "final" or model_name.startswith("checkpoint-"):
        model_name = os.path.basename(os.path.dirname(abs_path))

    return os.path.join("outputs", "evaluation", model_name, "preference_pairs.jsonl")

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


def calculate_text_perplexities(texts, model, tokenizer, batch_size):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    perplexities = []
    for _, batch in batched(texts, batch_size):
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().float()

        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.size())
        token_losses = token_losses * shift_mask
        seq_losses = token_losses.sum(dim=1) / shift_mask.sum(dim=1).clamp_min(1.0)
        perplexities.extend(torch.exp(seq_losses).cpu().tolist())

    return perplexities


def build_inputs(factual_texts, source_prefix):
    if source_prefix:
        return [source_prefix + text for text in factual_texts]
    return factual_texts


def batched(iterable, batch_size):
    for start_idx in range(0, len(iterable), batch_size):
        yield start_idx, iterable[start_idx:start_idx + batch_size]


def generate_predictions(model, tokenizer, factual_texts, source_prefix, device, batch_size, max_length, num_beams):
    model_inputs = build_inputs(factual_texts, source_prefix)
    generated_texts = []
    total_batches = (len(model_inputs) + batch_size - 1) // batch_size

    for _, batch in tqdm(batched(model_inputs, batch_size), total=total_batches, desc="Generating"):
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


def generate_candidate_predictions(
    model,
    tokenizer,
    factual_texts,
    source_prefix,
    device,
    batch_size,
    max_length,
    num_beams,
    num_candidates,
    candidate_strategy,
    temperature,
    top_p,
):
    model_inputs = build_inputs(factual_texts, source_prefix)
    all_candidates = []
    total_batches = (len(model_inputs) + batch_size - 1) // batch_size

    for _, batch in tqdm(batched(model_inputs, batch_size), total=total_batches, desc="Generating candidates"):
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(device)

        generation_kwargs = {
            "max_length": max_length,
            "num_return_sequences": num_candidates,
        }
        if candidate_strategy == "sample":
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "num_beams": 1,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
        else:
            generation_kwargs.update(
                {
                    "num_beams": max(num_beams, num_candidates),
                    "num_beam_groups": 1,
                }
            )

        with torch.no_grad():
            outputs = model.generate(**encoded, **generation_kwargs)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for offset in range(0, len(decoded), num_candidates):
            candidates = [text.strip() for text in decoded[offset:offset + num_candidates]]
            all_candidates.append(deduplicate_candidates(candidates))

    return all_candidates


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def tokenize_text(text):
    return re.findall(r"[a-z']+", normalize_text(text))


def deduplicate_candidates(candidates):
    seen = set()
    deduped = []
    for candidate in candidates:
        key = normalize_text(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped or [""]


def max_token_repeat_fraction(text):
    tokens = tokenize_text(text)
    if not tokens:
        return 0.0
    return max(tokens.count(token) for token in set(tokens)) / len(tokens)


def factual_similarity(factual_text, generated_text):
    return SequenceMatcher(None, normalize_text(factual_text), normalize_text(generated_text)).ratio()


def lexical_overlap_fraction(source_text, candidate_text):
    source_tokens = set(tokenize_text(source_text))
    candidate_tokens = tokenize_text(candidate_text)
    if not source_tokens or not candidate_tokens:
        return 0.0
    overlap = sum(token in source_tokens for token in candidate_tokens)
    return overlap / len(candidate_tokens)


def longest_shared_ngram_ratio(source_text, candidate_text, max_n=4):
    source_tokens = tokenize_text(source_text)
    candidate_tokens = tokenize_text(candidate_text)
    if len(source_tokens) < 2 or len(candidate_tokens) < 2:
        return 0.0

    max_shared = 0
    max_considered_n = min(max_n, len(source_tokens), len(candidate_tokens))
    for n in range(2, max_considered_n + 1):
        source_ngrams = {
            tuple(source_tokens[idx:idx + n])
            for idx in range(len(source_tokens) - n + 1)
        }
        if not source_ngrams:
            continue
        for idx in range(len(candidate_tokens) - n + 1):
            if tuple(candidate_tokens[idx:idx + n]) in source_ngrams:
                max_shared = n
    return max_shared / max(1, min(len(source_tokens), len(candidate_tokens), max_n))


def copy_penalty(source_text, candidate_text):
    exact_copy = float(normalize_text(source_text) == normalize_text(candidate_text))
    sequence_similarity = factual_similarity(source_text, candidate_text)
    lexical_overlap = lexical_overlap_fraction(source_text, candidate_text)
    shared_ngram_ratio = longest_shared_ngram_ratio(source_text, candidate_text)
    return {
        "exact_copy": exact_copy,
        "sequence_similarity": sequence_similarity,
        "lexical_overlap": lexical_overlap,
        "shared_ngram_ratio": shared_ngram_ratio,
        "copy_penalty": (
            1.5 * exact_copy
            + 0.5 * sequence_similarity
            + 0.3 * lexical_overlap
            + 0.2 * shared_ngram_ratio
        ),
    }


def opening_template(text):
    tokens = tokenize_text(text)
    if len(tokens) < 2:
        return ""
    return " ".join(tokens[:2])


def analyze_generation_failures(factual_texts, generated_texts):
    similarity_scores = [
        factual_similarity(factual, generated)
        for factual, generated in zip(factual_texts, generated_texts)
    ]
    repeat_fractions = [max_token_repeat_fraction(text) for text in generated_texts]
    opening_templates = [opening_template(text) for text in generated_texts]
    template_counter = Counter(template for template in opening_templates if template)

    exact_copy_mask = [
        normalize_text(factual) == normalize_text(generated)
        for factual, generated in zip(factual_texts, generated_texts)
    ]
    near_copy_mask = [score >= 0.9 for score in similarity_scores]
    high_repeat_mask = [fraction >= 0.35 for fraction in repeat_fractions]

    return {
        "similarity_scores": similarity_scores,
        "repeat_fractions": repeat_fractions,
        "opening_templates": opening_templates,
        "template_counter": template_counter,
        "exact_copy_mask": exact_copy_mask,
        "near_copy_mask": near_copy_mask,
        "high_repeat_mask": high_repeat_mask,
        "exact_copy_rate": float(np.mean(exact_copy_mask)),
        "near_copy_rate": float(np.mean(near_copy_mask)),
        "avg_similarity_to_factual": float(np.mean(similarity_scores)),
        "median_similarity_to_factual": float(np.median(similarity_scores)),
        "avg_max_token_repeat_fraction": float(np.mean(repeat_fractions)),
        "high_repeat_rate": float(np.mean(high_repeat_mask)),
    }


def score_candidates(
    factual_texts,
    candidate_groups,
    sarcasm_clf,
    sbert_model,
    gpt2_model,
    gpt2_tokenizer,
    batch_size,
    style_weight,
    similarity_weight,
    copy_weight,
    fluency_weight,
):
    flat_candidates = []
    candidate_index = []
    for example_idx, candidates in enumerate(candidate_groups):
        for candidate in candidates:
            flat_candidates.append(candidate)
            candidate_index.append(example_idx)

    sarcasm_results = sarcasm_clf(flat_candidates, batch_size=batch_size)
    sarcasm_scores = [
        result["score"] if result["label"].lower() == "sarcastic" else 1 - result["score"]
        for result in sarcasm_results
    ]
    fluency_perplexities = calculate_text_perplexities(
        flat_candidates,
        gpt2_model,
        gpt2_tokenizer,
        batch_size=batch_size,
    )

    source_embeddings = sbert_model.encode(factual_texts, convert_to_tensor=True, show_progress_bar=True)
    candidate_embeddings = sbert_model.encode(flat_candidates, convert_to_tensor=True, show_progress_bar=True)

    grouped = [[] for _ in factual_texts]
    for flat_idx, candidate in enumerate(flat_candidates):
        example_idx = candidate_index[flat_idx]
        similarity_score = util.cos_sim(
            source_embeddings[example_idx].unsqueeze(0),
            candidate_embeddings[flat_idx].unsqueeze(0),
        ).item()
        copy_metrics = copy_penalty(factual_texts[example_idx], candidate)
        fluency_penalty = float(np.log(max(fluency_perplexities[flat_idx], 1e-8)))
        total_score = (
            style_weight * sarcasm_scores[flat_idx]
            + similarity_weight * similarity_score
            - copy_weight * copy_metrics["copy_penalty"]
            - fluency_weight * fluency_penalty
        )
        grouped[example_idx].append(
            {
                "candidate_text": candidate,
                "sarcasm_score": sarcasm_scores[flat_idx],
                "semantic_similarity": similarity_score,
                "fluency_perplexity": fluency_perplexities[flat_idx],
                "fluency_penalty": fluency_penalty,
                **copy_metrics,
                "rerank_score": total_score,
            }
        )

    best_candidates = []
    for candidates in grouped:
        ranked = sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)
        best_candidates.append(ranked[0])

    return best_candidates, grouped


def build_preference_pairs(
    factual_texts,
    reference_texts,
    candidate_groups,
    source_prefix,
    pair_mode,
    min_score_margin,
):
    preference_pairs = []

    for example_idx, candidates in enumerate(candidate_groups):
        ranked = sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)
        if len(ranked) < 2:
            continue

        prompt_text = factual_texts[example_idx]
        prompt_with_prefix = (source_prefix or "") + prompt_text

        if pair_mode == "best_vs_worst":
            pair_indices = [(0, len(ranked) - 1)]
        else:
            pair_indices = [
                (chosen_idx, rejected_idx)
                for chosen_idx in range(len(ranked))
                for rejected_idx in range(chosen_idx + 1, len(ranked))
            ]

        for chosen_idx, rejected_idx in pair_indices:
            chosen = ranked[chosen_idx]
            rejected = ranked[rejected_idx]
            if chosen["candidate_text"] == rejected["candidate_text"]:
                continue
            score_margin = chosen["rerank_score"] - rejected["rerank_score"]
            if score_margin < min_score_margin:
                continue

            preference_pairs.append(
                {
                    "example_idx": example_idx,
                    "prompt": prompt_text,
                    "prompt_with_prefix": prompt_with_prefix,
                    "reference_satirical": reference_texts[example_idx],
                    "chosen": chosen["candidate_text"],
                    "rejected": rejected["candidate_text"],
                    "score_margin": score_margin,
                    "chosen_score": chosen["rerank_score"],
                    "rejected_score": rejected["rerank_score"],
                    "chosen_sarcasm": chosen["sarcasm_score"],
                    "rejected_sarcasm": rejected["sarcasm_score"],
                    "chosen_similarity": chosen["semantic_similarity"],
                    "rejected_similarity": rejected["semantic_similarity"],
                    "chosen_copy_penalty": chosen["copy_penalty"],
                    "rejected_copy_penalty": rejected["copy_penalty"],
                    "pair_mode": pair_mode,
                }
            )

    return preference_pairs


def save_preference_pairs(preference_pairs, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for pair in preference_pairs:
            handle.write(json.dumps(pair, ensure_ascii=True) + "\n")


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

    if args.num_candidates > 1:
        print("Generating and reranking candidates...")
        candidate_groups = generate_candidate_predictions(
            model=model,
            tokenizer=tokenizer,
            factual_texts=factual_texts,
            source_prefix=source_prefix,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_beams=args.num_beams,
            num_candidates=args.num_candidates,
            candidate_strategy=args.candidate_strategy,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        best_candidates, all_candidate_scores = score_candidates(
            factual_texts=factual_texts,
            candidate_groups=candidate_groups,
            sarcasm_clf=sarcasm_clf,
            sbert_model=sbert_model,
            gpt2_model=gpt2_model,
            gpt2_tokenizer=gpt2_tokenizer,
            batch_size=args.batch_size,
            style_weight=args.style_weight,
            similarity_weight=args.similarity_weight,
            copy_weight=args.copy_weight,
            fluency_weight=args.fluency_weight,
        )
        generated_texts = [candidate["candidate_text"] for candidate in best_candidates]
        sarcasm_scores = [candidate["sarcasm_score"] for candidate in best_candidates]
        avg_sarcasm = float(np.mean(sarcasm_scores))
        avg_cosine = float(np.mean([candidate["semantic_similarity"] for candidate in best_candidates]))
    else:
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
        all_candidate_scores = None

    ppl = calculate_perplexity(generated_texts, gpt2_model, gpt2_tokenizer)
    failure_analysis = analyze_generation_failures(factual_texts, generated_texts)

    results_payload = {
        "factual": factual_texts,
        "reference_satirical": reference_texts,
        "generated_satirical": generated_texts,
        "sarcasm_score": sarcasm_scores,
        "similarity_to_factual": failure_analysis["similarity_scores"],
        "max_token_repeat_fraction": failure_analysis["repeat_fractions"],
        "opening_template": failure_analysis["opening_templates"],
        "is_exact_copy": failure_analysis["exact_copy_mask"],
        "is_near_copy": failure_analysis["near_copy_mask"],
        "is_high_repeat": failure_analysis["high_repeat_mask"],
    }
    if all_candidate_scores is not None:
        results_payload["selected_semantic_similarity"] = [
            candidate["semantic_similarity"] for candidate in best_candidates
        ]
        results_payload["selected_copy_penalty"] = [
            candidate["copy_penalty"] for candidate in best_candidates
        ]
        results_payload["selected_rerank_score"] = [
            candidate["rerank_score"] for candidate in best_candidates
        ]
        results_payload["selected_fluency_perplexity"] = [
            candidate["fluency_perplexity"] for candidate in best_candidates
        ]
        results_payload["candidate_details_json"] = [
            json.dumps(candidates, ensure_ascii=True) for candidates in all_candidate_scores
        ]
    results_df = pd.DataFrame(results_payload)
    if not args.no_save_csv:
        output_csv = args.output_csv or default_output_csv(model_path)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        print(f"Saved detailed predictions to: {output_csv}")

    if all_candidate_scores is not None and args.save_preference_pairs:
        preference_pairs = build_preference_pairs(
            factual_texts=factual_texts,
            reference_texts=reference_texts,
            candidate_groups=all_candidate_scores,
            source_prefix=source_prefix,
            pair_mode=args.preference_pair_mode,
            min_score_margin=args.preference_min_margin,
        )
        preference_output = args.preference_output_jsonl or default_preference_jsonl(model_path)
        save_preference_pairs(preference_pairs, preference_output)
        print(f"Saved {len(preference_pairs)} preference pairs to: {preference_output}")

    print("\n--- Evaluation Results ---")
    print(f"Num test examples                       : {len(test_df)}")
    print(f"Style Accuracy (RoBERTa sarcasm prob)   : {avg_sarcasm:.4f}")
    print(f"Content Preservation (SBERT cosine)     : {avg_cosine:.4f}")
    print(f"Linguistic Fluency (GPT-2 perplexity)   : {ppl:.2f}")
    print(f"Exact Copy Rate                         : {failure_analysis['exact_copy_rate']:.4f}")
    print(f"Near-Copy Rate (sim >= 0.90)            : {failure_analysis['near_copy_rate']:.4f}")
    print(f"Avg Similarity To Factual               : {failure_analysis['avg_similarity_to_factual']:.4f}")
    print(f"Median Similarity To Factual            : {failure_analysis['median_similarity_to_factual']:.4f}")
    print(f"High Repetition Rate                    : {failure_analysis['high_repeat_rate']:.4f}")
    print(f"Avg Max Token Repeat Fraction           : {failure_analysis['avg_max_token_repeat_fraction']:.4f}")
    if args.num_candidates > 1:
        print(f"Candidate Count                         : {args.num_candidates}")
        print(f"Candidate Strategy                      : {args.candidate_strategy}")
        print(
            "Reranker Weights                       : "
            f"style={args.style_weight:.2f}, "
            f"similarity={args.similarity_weight:.2f}, "
            f"copy={args.copy_weight:.2f}, "
            f"fluency={args.fluency_weight:.2f}"
        )
        if args.save_preference_pairs:
            print(f"Preference Pair Mode                   : {args.preference_pair_mode}")
            print(f"Preference Min Margin                  : {args.preference_min_margin:.4f}")
    print("\nTop Opening Templates:")
    for template, count in failure_analysis["template_counter"].most_common(10):
        print(f"  {count:>4}  {template}")

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
    parser.add_argument("--num_candidates", type=int, default=1, help="How many candidates to generate per input before reranking.")
    parser.add_argument("--candidate_strategy", type=str, choices=["beam", "sample"], default="beam", help="How to generate multiple candidates.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature when candidate_strategy=sample.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p nucleus sampling threshold when candidate_strategy=sample.")
    parser.add_argument("--style_weight", type=float, default=2.0, help="Weight for sarcasm score in candidate reranking.")
    parser.add_argument("--similarity_weight", type=float, default=1.0, help="Weight for SBERT similarity in candidate reranking.")
    parser.add_argument("--copy_weight", type=float, default=1.5, help="Penalty weight for copy-heavy candidates in reranking.")
    parser.add_argument("--fluency_weight", type=float, default=0.15, help="Penalty weight for GPT-2 log-perplexity in candidate reranking.")
    parser.add_argument("--save_preference_pairs", action="store_true", help="Save chosen/rejected preference pairs for DPO-style training. Requires --num_candidates > 1.")
    parser.add_argument("--preference_output_jsonl", type=str, default=None, help="Optional JSONL path for saved preference pairs.")
    parser.add_argument("--preference_pair_mode", type=str, choices=["best_vs_worst", "all_pairs"], default="best_vs_worst", help="Whether to save only the top-vs-bottom pair or all ordered pairs per example.")
    parser.add_argument("--preference_min_margin", type=float, default=0.05, help="Minimum reranker score gap required before saving a preference pair.")
    parser.add_argument("--num_examples", type=int, default=5, help="How many random example predictions to print.")
    parser.add_argument("--source_prefix", type=str, default=None, help="Optional task prefix. Leave unset for BART; set for T5 if needed.")
    args = parser.parse_args()
    if args.save_preference_pairs and args.num_candidates <= 1:
        parser.error("--save_preference_pairs requires --num_candidates > 1.")
    main(args)
