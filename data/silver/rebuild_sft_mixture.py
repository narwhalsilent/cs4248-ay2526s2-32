import argparse
import csv
import json
import math
import random
import re
from collections import Counter
from pathlib import Path


DEFAULT_SILVER_PATH = "data/silver/silver_headlines.jsonl"
DEFAULT_DESARCASTIC_PATH = "data/silver/desarcastic_headlines_complete.txt"
DEFAULT_OUTPUT_DIR = "data/silver/rebalanced"
DEFAULT_SEED = 42
DEFAULT_ORIGINAL_TO_DESARCASTIC_RATIO = 1.0
DEFAULT_MIN_NGRAM_COUNT = 75
DEFAULT_NGRAM_SIZES = (2, 3)
DEFAULT_MIN_DESARCASTIC_CONFIDENCE = 0.1
DEFAULT_MIN_DESARCASTIC_CONTENT = 0.08
DEFAULT_MIN_SILVER_CONFIDENCE = 0.0


def normalize_text(text):
    return re.sub(r"\s+", " ", text.strip())


def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


def extract_ngrams(text, ngram_sizes):
    tokens = tokenize(text)
    grams = set()
    for n in ngram_sizes:
        for idx in range(len(tokens) - n + 1):
            grams.add(" ".join(tokens[idx : idx + n]))
    return grams


def load_original_silver(path, min_confidence):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            confidence = float(item.get("confidence_score", 0.0))
            if confidence < min_confidence:
                continue
            factual = normalize_text(item["factual_headline"])
            satirical = normalize_text(item["silver_sarcastic_headline"])
            if not factual or not satirical:
                continue
            records.append(
                {
                    "factual": factual,
                    "satirical": satirical,
                    "source": "silver",
                    "confidence": confidence,
                }
            )
    return records


def load_desarcastic(path, min_confidence, min_content):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            confidence = float(item.get("confidence_score", 0.0))
            content_score = float(item.get("content_score", 0.0))
            if confidence < min_confidence:
                continue
            if content_score < min_content:
                continue
            factual = normalize_text(item["silver_factual_headline"])
            satirical = normalize_text(item["original_sarcastic_headline"])
            if not factual or not satirical:
                continue
            records.append(
                {
                    "factual": factual,
                    "satirical": satirical,
                    "source": "desarcastic",
                    "confidence": confidence,
                    "content_score": content_score,
                }
            )
    return records


def compute_overused_phrases(records, min_ngram_count, ngram_sizes):
    counts = Counter()
    for record in records:
        for gram in extract_ngrams(record["satirical"], ngram_sizes):
            counts[gram] += 1
    overused = {gram: count for gram, count in counts.items() if count >= min_ngram_count}
    return counts, overused


def score_original_records(records, overused_phrases, ngram_sizes):
    scored = []
    for record in records:
        grams = extract_ngrams(record["satirical"], ngram_sizes)
        hits = {gram: overused_phrases[gram] for gram in grams if gram in overused_phrases}
        repetition_penalty = sum(math.log1p(count) for count in hits.values())
        weight = 1.0 / (1.0 + repetition_penalty)
        scored.append(
            {
                **record,
                "repetition_penalty": repetition_penalty,
                "weight": weight,
                "overused_phrases": sorted(hits.items(), key=lambda item: item[1], reverse=True),
            }
        )
    return scored


def weighted_sample_without_replacement(records, sample_size, seed):
    rng = random.Random(seed)
    pool = list(records)
    chosen = []
    target = min(sample_size, len(pool))

    while pool and len(chosen) < target:
        total_weight = sum(max(item["weight"], 1e-8) for item in pool)
        pick = rng.random() * total_weight
        cumulative = 0.0
        selected_idx = len(pool) - 1
        for idx, item in enumerate(pool):
            cumulative += max(item["weight"], 1e-8)
            if cumulative >= pick:
                selected_idx = idx
                break
        chosen.append(pool.pop(selected_idx))

    return chosen


def dedupe_records(records):
    deduped = []
    seen = set()
    for record in records:
        key = (record["factual"].lower(), record["satirical"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def split_records(records, seed):
    rng = random.Random(seed)
    items = list(records)
    rng.shuffle(items)
    total = len(items)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }


def combine_source_splits(original_records, desarcastic_records, seed):
    original_splits = split_records(original_records, seed)
    desarcastic_splits = split_records(desarcastic_records, seed + 1)
    combined = {}
    rng = random.Random(seed + 2)
    for split_name in ("train", "val", "test"):
        merged = original_splits[split_name] + desarcastic_splits[split_name]
        rng.shuffle(merged)
        combined[split_name] = merged
    return combined


def write_csv(path, records):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["factual", "satirical"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "factual": record["factual"],
                    "satirical": record["satirical"],
                }
            )


def write_metadata(path, stats):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def build_dataset(args):
    original_records = load_original_silver(args.silver_path, args.min_silver_confidence)
    desarcastic_records = load_desarcastic(
        args.desarcastic_path,
        args.min_desarcastic_confidence,
        args.min_desarcastic_content,
    )

    original_counts, overused = compute_overused_phrases(
        original_records,
        min_ngram_count=args.min_ngram_count,
        ngram_sizes=args.ngram_sizes,
    )
    scored_original = score_original_records(
        original_records,
        overused_phrases=overused,
        ngram_sizes=args.ngram_sizes,
    )

    target_original = int(len(desarcastic_records) * args.original_to_desarcastic_ratio)
    sampled_original = weighted_sample_without_replacement(
        scored_original,
        sample_size=target_original,
        seed=args.seed,
    )

    sampled_original = dedupe_records(sampled_original)
    desarcastic_records = dedupe_records(desarcastic_records)
    combined_splits = combine_source_splits(
        sampled_original,
        desarcastic_records,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, records in combined_splits.items():
        write_csv(output_dir / f"{split_name}.csv", records)

    stats = {
        "silver_path": args.silver_path,
        "desarcastic_path": args.desarcastic_path,
        "seed": args.seed,
        "original_to_desarcastic_ratio": args.original_to_desarcastic_ratio,
        "min_ngram_count": args.min_ngram_count,
        "ngram_sizes": list(args.ngram_sizes),
        "min_silver_confidence": args.min_silver_confidence,
        "min_desarcastic_confidence": args.min_desarcastic_confidence,
        "min_desarcastic_content": args.min_desarcastic_content,
        "original_loaded": len(original_records),
        "desarcastic_loaded": len(desarcastic_records),
        "original_sampled": len(sampled_original),
        "overused_phrase_count": len(overused),
        "top_overused_phrases": original_counts.most_common(50),
        "split_sizes": {name: len(records) for name, records in combined_splits.items()},
        "source_breakdown": {
            name: {
                "silver": sum(record["source"] == "silver" for record in records),
                "desarcastic": sum(record["source"] == "desarcastic" for record in records),
            }
            for name, records in combined_splits.items()
        },
        "top_downweighted_examples": [
            {
                "factual": record["factual"],
                "satirical": record["satirical"],
                "weight": round(record["weight"], 6),
                "repetition_penalty": round(record["repetition_penalty"], 4),
                "overused_phrases": record["overused_phrases"][:10],
            }
            for record in sorted(
                scored_original,
                key=lambda item: item["repetition_penalty"],
                reverse=True,
            )[:25]
        ],
    }
    write_metadata(output_dir / "mixture_stats.json", stats)

    print(f"Wrote mixed SFT dataset to {output_dir}")
    print(f"Split sizes: {stats['split_sizes']}")
    print(f"Source breakdown: {stats['source_breakdown']}")
    print("Top overused phrases:")
    for phrase, count in stats["top_overused_phrases"][:15]:
        print(f"  {count:>5}  {phrase}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild SFT splits by mixing desarcastic pairs and downweighting templatic silver pairs."
    )
    parser.add_argument("--silver-path", default=DEFAULT_SILVER_PATH)
    parser.add_argument("--desarcastic-path", default=DEFAULT_DESARCASTIC_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--original-to-desarcastic-ratio",
        type=float,
        default=DEFAULT_ORIGINAL_TO_DESARCASTIC_RATIO,
    )
    parser.add_argument("--min-ngram-count", type=int, default=DEFAULT_MIN_NGRAM_COUNT)
    parser.add_argument(
        "--ngram-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_NGRAM_SIZES),
    )
    parser.add_argument(
        "--min-desarcastic-confidence",
        type=float,
        default=DEFAULT_MIN_DESARCASTIC_CONFIDENCE,
    )
    parser.add_argument(
        "--min-desarcastic-content",
        type=float,
        default=DEFAULT_MIN_DESARCASTIC_CONTENT,
    )
    parser.add_argument(
        "--min-silver-confidence",
        type=float,
        default=DEFAULT_MIN_SILVER_CONFIDENCE,
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
