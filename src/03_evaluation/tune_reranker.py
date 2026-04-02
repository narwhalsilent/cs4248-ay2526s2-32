import argparse
import itertools
import json
import os
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def build_inputs(factual_texts, source_prefix):
    if source_prefix:
        return [source_prefix + text for text in factual_texts]
    return factual_texts


def batched(iterable, batch_size):
    for start_idx in range(0, len(iterable), batch_size):
        yield start_idx, iterable[start_idx:start_idx + batch_size]


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

    for _, batch in batched(model_inputs, batch_size):
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
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
                }
            )

        with torch.no_grad():
            outputs = model.generate(**encoded, **generation_kwargs)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for offset in range(0, len(decoded), num_candidates):
            candidates = [text.strip() for text in decoded[offset:offset + num_candidates]]
            all_candidates.append(deduplicate_candidates(candidates))

    return all_candidates


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
        penalty = copy_penalty(factual_texts[example_idx], candidate)
        fluency_penalty = float(np.log(max(fluency_perplexities[flat_idx], 1e-8)))
        rerank_score = (
            style_weight * sarcasm_scores[flat_idx]
            + similarity_weight * similarity_score
            - copy_weight * penalty["copy_penalty"]
            - fluency_weight * fluency_penalty
        )
        grouped[example_idx].append(
            {
                "candidate_text": candidate,
                "sarcasm_score": sarcasm_scores[flat_idx],
                "semantic_similarity": similarity_score,
                "fluency_perplexity": fluency_perplexities[flat_idx],
                "fluency_penalty": fluency_penalty,
                **penalty,
                "rerank_score": rerank_score,
            }
        )
    return grouped


def summarize_selection(factual_texts, scored_candidates, min_margin):
    chosen = [max(candidates, key=lambda item: item["rerank_score"]) for candidates in scored_candidates]
    exact_copy_rate = float(np.mean([candidate["exact_copy"] for candidate in chosen]))
    near_copy_rate = float(np.mean([candidate["sequence_similarity"] >= 0.90 for candidate in chosen]))
    avg_sarcasm = float(np.mean([candidate["sarcasm_score"] for candidate in chosen]))
    avg_similarity = float(np.mean([candidate["semantic_similarity"] for candidate in chosen]))
    avg_copy_penalty = float(np.mean([candidate["copy_penalty"] for candidate in chosen]))
    avg_fluency_perplexity = float(np.mean([candidate["fluency_perplexity"] for candidate in chosen]))

    pair_count = 0
    for candidates in scored_candidates:
        ranked = sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)
        for chosen_idx in range(len(ranked)):
            for rejected_idx in range(chosen_idx + 1, len(ranked)):
                if ranked[chosen_idx]["rerank_score"] - ranked[rejected_idx]["rerank_score"] >= min_margin:
                    pair_count += 1

    selection_score = 2.0 * avg_sarcasm + avg_similarity - exact_copy_rate - near_copy_rate
    return {
        "avg_sarcasm": avg_sarcasm,
        "avg_similarity": avg_similarity,
        "avg_copy_penalty": avg_copy_penalty,
        "avg_fluency_perplexity": avg_fluency_perplexity,
        "exact_copy_rate": exact_copy_rate,
        "near_copy_rate": near_copy_rate,
        "preference_pair_count": pair_count,
        "selection_score": selection_score,
    }


def parse_list(value, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    df = pd.read_csv(args.validation_data)
    factual_texts = df["factual"].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)
    model.eval()

    source_prefix = args.source_prefix
    if source_prefix is None and "t5" in os.path.basename(args.model_path).lower():
        source_prefix = "translate factual to satire: "
    source_prefix = source_prefix or ""

    sarcasm_clf = pipeline(
        "text-classification",
        model="helinivan/english-sarcasm-detector",
        device=0 if device == "cuda" else -1
    )
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    gpt2_model.eval()

    candidate_counts = parse_list(args.num_candidates_grid, int)
    strategies = parse_list(args.strategy_grid, str)
    style_weights = parse_list(args.style_weight_grid, float)
    similarity_weights = parse_list(args.similarity_weight_grid, float)
    copy_weights = parse_list(args.copy_weight_grid, float)
    fluency_weights = parse_list(args.fluency_weight_grid, float)
    margins = parse_list(args.margin_grid, float)

    rows = []
    for strategy, num_candidates in itertools.product(strategies, candidate_counts):
        candidate_groups = generate_candidate_predictions(
            model=model,
            tokenizer=tokenizer,
            factual_texts=factual_texts,
            source_prefix=source_prefix,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_beams=args.num_beams,
            num_candidates=num_candidates,
            candidate_strategy=strategy,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        for style_weight, similarity_weight, copy_weight, fluency_weight in itertools.product(
            style_weights,
            similarity_weights,
            copy_weights,
            fluency_weights,
        ):
            scored = score_candidates(
                factual_texts=factual_texts,
                candidate_groups=candidate_groups,
                sarcasm_clf=sarcasm_clf,
                sbert_model=sbert_model,
                gpt2_model=gpt2_model,
                gpt2_tokenizer=gpt2_tokenizer,
                batch_size=args.batch_size,
                style_weight=style_weight,
                similarity_weight=similarity_weight,
                copy_weight=copy_weight,
                fluency_weight=fluency_weight,
            )
            for margin in margins:
                summary = summarize_selection(factual_texts, scored, margin)
                rows.append(
                    {
                        "candidate_strategy": strategy,
                        "num_candidates": num_candidates,
                        "style_weight": style_weight,
                        "similarity_weight": similarity_weight,
                        "copy_weight": copy_weight,
                        "fluency_weight": fluency_weight,
                        "preference_min_margin": margin,
                        **summary,
                    }
                )

    results_df = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Saved reranker sweep to: {args.output_csv}")

    best = results_df.iloc[0].to_dict()
    if args.best_config_json:
        os.makedirs(os.path.dirname(args.best_config_json), exist_ok=True)
        with open(args.best_config_json, "w", encoding="utf-8") as handle:
            json.dump(best, handle, indent=2)
        print(f"Saved best config to: {args.best_config_json}")

    print("Best configuration:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep reranker weights and decoding settings on the validation split.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--validation_data", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="outputs/evaluation/reranker_sweep.csv")
    parser.add_argument("--best_config_json", type=str, default="outputs/evaluation/best_reranker_config.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_candidates_grid", type=str, default="4,8")
    parser.add_argument("--strategy_grid", type=str, default="sample,beam")
    parser.add_argument("--style_weight_grid", type=str, default="1.5,2.0,2.5")
    parser.add_argument("--similarity_weight_grid", type=str, default="0.75,1.0,1.25")
    parser.add_argument("--copy_weight_grid", type=str, default="1.0,1.5,2.0")
    parser.add_argument("--fluency_weight_grid", type=str, default="0.0,0.1,0.15")
    parser.add_argument("--margin_grid", type=str, default="0.05,0.1,0.2")
    parser.add_argument("--source_prefix", type=str, default=None)
    args = parser.parse_args()
    main(args)
