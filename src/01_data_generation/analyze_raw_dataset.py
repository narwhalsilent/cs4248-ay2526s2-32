"""
Analyze the raw sarcasm headlines dataset.

This mirrors the silver-data analysis at the headline level, but for the
original JSONL dataset and without producing CSV artifacts.

Usage:
    python3 src/01_data_generation/analyze_raw_dataset.py
    python3 src/01_data_generation/analyze_raw_dataset.py --input data/raw/Sarcasm_Headlines_Dataset_v2.json --top_k 20
"""

import argparse
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

WORD_RE = re.compile(r"[a-z0-9']+")


def normalize_text(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text):
    return WORD_RE.findall(normalize_text(text))


def safe_pct(part, whole):
    if whole == 0:
        return 0.0
    return 100.0 * float(part) / float(whole)


def prefix_counter(texts, n):
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        if not tokens:
            continue
        prefix = " ".join(tokens[:n])
        counter[prefix] += 1
    return counter


def format_top(counter, total_rows, top_k):
    lines = []
    for phrase, count in counter.most_common(top_k):
        lines.append((phrase, count, safe_pct(count, total_rows)))
    return lines


def extract_domain(url):
    text = normalize_text(url)
    text = re.sub(r"^https?://", "", text)
    return text.split("/", 1)[0]


def add_headline_section(lines, title, headlines, top_k):
    total = len(headlines)
    normalized = [normalize_text(t) for t in headlines]
    exact_unique = len(set(headlines))
    norm_unique = len(set(normalized))

    tokenized = [tokenize(t) for t in headlines]
    token_count = sum(len(tokens) for tokens in tokenized)
    vocab = set(token for tokens in tokenized for token in tokens)

    prefix2 = prefix_counter(headlines, n=2)
    top_2 = format_top(prefix2, total, top_k)
    top1_share_2 = safe_pct(prefix2.most_common(1)[0][1], total) if prefix2 else 0.0
    top5_share_2 = safe_pct(sum(c for _, c in prefix2.most_common(5)), total) if prefix2 else 0.0

    template_markers = [
        "nation horrified",
        "new study reveals",
        "local man",
        "area man",
        "experts baffled",
        "white house",
    ]
    marker_counts = []
    for marker in template_markers:
        count = sum(1 for s in normalized if marker in s)
        marker_counts.append((marker, count, safe_pct(count, total)))

    avg_len = float(np.mean([len(tokens) for tokens in tokenized])) if tokenized else 0.0
    std_len = float(np.std([len(tokens) for tokens in tokenized])) if tokenized else 0.0
    ttr = float(len(vocab) / token_count) if token_count else 0.0

    lines.append(f"## {title}")
    lines.append(f"- Rows: {total}")
    lines.append(f"- Exact duplicate headlines: {total - exact_unique} ({safe_pct(total - exact_unique, total):.2f}%)")
    lines.append(f"- Normalized duplicate headlines: {total - norm_unique} ({safe_pct(total - norm_unique, total):.2f}%)")
    lines.append(f"- Avg headline length (tokens): {avg_len:.2f}")
    lines.append(f"- Std headline length (tokens): {std_len:.2f}")
    lines.append(f"- Vocabulary size: {len(vocab)}")
    lines.append(f"- Type-token ratio: {ttr:.4f}")
    lines.append(f"- Share of most common 2-word prefix: {top1_share_2:.2f}%")
    lines.append(f"- Share covered by top 5 2-word prefixes: {top5_share_2:.2f}%")
    lines.append("")

    lines.append("Top 2-word prefixes:")
    lines.append("")
    lines.append("| Prefix | Count | Percent |")
    lines.append("|---|---:|---:|")
    for phrase, count, pct in top_2:
        safe_phrase = phrase.replace("|", "\\|")
        lines.append(f"| {safe_phrase} | {count} | {pct:.2f}% |")
    lines.append("")

    lines.append("Template marker frequencies:")
    lines.append("")
    lines.append("| Marker | Count | Percent |")
    lines.append("|---|---:|---:|")
    for marker, count, pct in marker_counts:
        lines.append(f"| {marker} | {count} | {pct:.2f}% |")
    lines.append("")


def main(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_json(args.input, lines=True)
    required = {"headline", "is_sarcastic", "article_link"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"JSON is missing required fields: {sorted(missing)}")

    before = len(df)
    df = df.dropna(subset=["headline", "is_sarcastic", "article_link"]).copy()
    df["headline"] = df["headline"].astype(str)
    df["article_link"] = df["article_link"].astype(str)
    df = df[df["headline"].str.strip() != ""].copy()
    df["is_sarcastic"] = df["is_sarcastic"].astype(int)

    after = len(df)
    headlines = df["headline"].tolist()
    sarcastic = df[df["is_sarcastic"] == 1]["headline"].tolist()
    non_sarcastic = df[df["is_sarcastic"] == 0]["headline"].tolist()

    domains = df["article_link"].map(extract_domain)
    domain_counts = Counter(domains)
    label_counts = df["is_sarcastic"].value_counts().to_dict()

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    report_path = os.path.join(args.output_dir, f"{base_name}_raw_quality_report.md")

    lines = []
    lines.append(f"# Raw Dataset Quality Report: {args.input}")
    lines.append("")
    lines.append("## 1) Basic Stats")
    lines.append(f"- Rows loaded: {before}")
    lines.append(f"- Rows after cleaning: {after}")
    lines.append(f"- Sarcastic headlines: {label_counts.get(1, 0)} ({safe_pct(label_counts.get(1, 0), after):.2f}%)")
    lines.append(f"- Non-sarcastic headlines: {label_counts.get(0, 0)} ({safe_pct(label_counts.get(0, 0), after):.2f}%)")
    lines.append("")

    lines.append("## 2) Source Domains")
    lines.append("| Domain | Count | Percent |")
    lines.append("|---|---:|---:|")
    for domain, count in domain_counts.most_common(args.top_k):
        lines.append(f"| {domain} | {count} | {safe_pct(count, after):.2f}% |")
    lines.append("")

    add_headline_section(lines, "3) All Headlines", headlines, args.top_k)
    add_headline_section(lines, "4) Sarcastic Headlines Only", sarcastic, args.top_k)
    add_headline_section(lines, "5) Non-Sarcastic Headlines Only", non_sarcastic, args.top_k)

    lines.append("## 6) Interpretation Guide")
    lines.append("- High prefix concentration suggests repeated framing patterns.")
    lines.append("- Duplicate rates reveal how much headline reuse exists in the raw data.")
    lines.append("- Comparing sarcastic vs non-sarcastic subsets helps show whether one class is more templated or lexically narrow.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Raw dataset analysis complete.")
    print(f"Input rows (cleaned): {after}")
    print(f"Sarcastic rows: {label_counts.get(1, 0)}")
    print(f"Non-sarcastic rows: {label_counts.get(0, 0)}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze raw sarcasm headline dataset quality and template concentration.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/Sarcasm_Headlines_Dataset_v2.json",
        help="Input JSONL path with headline, is_sarcastic, and article_link fields.",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/analysis", help="Directory for Markdown report.")
    parser.add_argument("--top_k", type=int, default=15, help="How many top prefixes/domains to report.")
    args = parser.parse_args()
    main(args)
