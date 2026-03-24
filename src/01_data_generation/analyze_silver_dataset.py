"""
Analyze silver dataset quality and stylistic diversity.

This script highlights signs of weak supervision such as:
- template concentration (e.g., many targets starting with the same phrase)
- low lexical diversity
- high factual-to-satirical overlap (weak rewriting)

Usage:
    python src/01_data_generation/analyze_silver_dataset.py
    python src/01_data_generation/analyze_silver_dataset.py --input data/silver/train.csv --top_k 20
"""

import argparse
import os
import re
from collections import Counter
from difflib import SequenceMatcher

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


def lexical_overlap_jaccard(src, tgt):
    src_set = set(tokenize(src))
    tgt_set = set(tokenize(tgt))
    if not src_set and not tgt_set:
        return 1.0
    if not src_set or not tgt_set:
        return 0.0
    return len(src_set & tgt_set) / len(src_set | tgt_set)


def seq_ratio(src, tgt):
    return SequenceMatcher(None, normalize_text(src), normalize_text(tgt)).ratio()


def format_top(counter, total_rows, top_k):
    lines = []
    for phrase, count in counter.most_common(top_k):
        lines.append((phrase, count, safe_pct(count, total_rows)))
    return lines


def main(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    required = {"factual", "satirical"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    before = len(df)
    df = df.dropna(subset=["factual", "satirical"]).copy()
    df["factual"] = df["factual"].astype(str)
    df["satirical"] = df["satirical"].astype(str)
    df = df[(df["factual"].str.strip() != "") & (df["satirical"].str.strip() != "")].copy()
    after = len(df)

    factual = df["factual"].tolist()
    satirical = df["satirical"].tolist()
    total = len(df)

    sat_norm = [normalize_text(t) for t in satirical]
    factual_norm = [normalize_text(t) for t in factual]

    exact_unique_sat = len(set(satirical))
    norm_unique_sat = len(set(sat_norm))

    tokenized_targets = [tokenize(t) for t in satirical]
    target_token_count = sum(len(toks) for toks in tokenized_targets)
    target_vocab = set(tok for toks in tokenized_targets for tok in toks)

    prefix2 = prefix_counter(satirical, n=2)
    prefix3 = prefix_counter(satirical, n=3)

    top_2 = format_top(prefix2, total, args.top_k)
    top_3 = format_top(prefix3, total, args.top_k)

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
        count = sum(1 for s in sat_norm if marker in s)
        marker_counts.append((marker, count, safe_pct(count, total)))

    jaccards = np.array([lexical_overlap_jaccard(s, t) for s, t in zip(factual, satirical)], dtype=float)
    seqs = np.array([seq_ratio(s, t) for s, t in zip(factual, satirical)], dtype=float)
    exact_copy_count = sum(1 for s, t in zip(factual_norm, sat_norm) if s == t)

    high_jaccard = int((jaccards >= args.jaccard_threshold).sum())
    high_seq = int((seqs >= args.seq_threshold).sum())

    avg_target_len = float(np.mean([len(t) for t in tokenized_targets])) if tokenized_targets else 0.0
    std_target_len = float(np.std([len(t) for t in tokenized_targets])) if tokenized_targets else 0.0
    ttr = float(len(target_vocab) / target_token_count) if target_token_count else 0.0

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    report_path = os.path.join(args.output_dir, f"{base_name}_quality_report.md")
    top2_path = os.path.join(args.output_dir, f"{base_name}_top_prefix2.csv")
    top3_path = os.path.join(args.output_dir, f"{base_name}_top_prefix3.csv")

    pd.DataFrame(top_2, columns=["prefix2", "count", "percent"]).to_csv(top2_path, index=False)
    pd.DataFrame(top_3, columns=["prefix3", "count", "percent"]).to_csv(top3_path, index=False)

    lines = []
    lines.append(f"# Silver Dataset Quality Report: {args.input}")
    lines.append("")
    lines.append("## 1) Basic Stats")
    lines.append(f"- Rows loaded: {before}")
    lines.append(f"- Rows after cleaning: {after}")
    lines.append(f"- Exact duplicate satirical headlines: {total - exact_unique_sat} ({safe_pct(total - exact_unique_sat, total):.2f}%)")
    lines.append(f"- Normalized duplicate satirical headlines: {total - norm_unique_sat} ({safe_pct(total - norm_unique_sat, total):.2f}%)")
    lines.append("")

    lines.append("## 2) Template Concentration")
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

    lines.append("## 3) Diversity and Length")
    lines.append(f"- Avg satirical length (tokens): {avg_target_len:.2f}")
    lines.append(f"- Std satirical length (tokens): {std_target_len:.2f}")
    lines.append(f"- Vocabulary size (satirical): {len(target_vocab)}")
    lines.append(f"- Type-token ratio (satirical): {ttr:.4f}")
    lines.append("")

    lines.append("## 4) Source-Target Overlap")
    lines.append(f"- Mean lexical Jaccard overlap: {float(jaccards.mean()):.4f}")
    lines.append(f"- Median lexical Jaccard overlap: {float(np.median(jaccards)):.4f}")
    lines.append(f"- Mean normalized sequence similarity: {float(seqs.mean()):.4f}")
    lines.append(f"- Median normalized sequence similarity: {float(np.median(seqs)):.4f}")
    lines.append(f"- Exact copies (source == target after normalization): {exact_copy_count} ({safe_pct(exact_copy_count, total):.2f}%)")
    lines.append(
        f"- High lexical overlap (Jaccard >= {args.jaccard_threshold:.2f}): "
        f"{high_jaccard} ({safe_pct(high_jaccard, total):.2f}%)"
    )
    lines.append(
        f"- High sequence similarity (ratio >= {args.seq_threshold:.2f}): "
        f"{high_seq} ({safe_pct(high_seq, total):.2f}%)"
    )
    lines.append("")

    lines.append("## 5) Interpretation Guide")
    lines.append("- High prefix concentration suggests style template collapse.")
    lines.append("- High source-target overlap suggests weak rewrites (too close to factual headlines).")
    lines.append("- Low diversity metrics suggest limited stylistic coverage for training.")
    lines.append("")
    lines.append("## 6) Artifacts")
    lines.append(f"- 2-word prefix table: {top2_path}")
    lines.append(f"- 3-word prefix table: {top3_path}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Silver dataset analysis complete.")
    print(f"Input rows (cleaned): {total}")
    print(f"Top 1 prefix share (2-word): {top1_share_2:.2f}%")
    print(f"Top 5 prefix share (2-word): {top5_share_2:.2f}%")
    print(f"Mean lexical Jaccard overlap: {float(jaccards.mean()):.4f}")
    print(f"Mean sequence similarity: {float(seqs.mean()):.4f}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze silver dataset quality and template concentration.")
    parser.add_argument("--input", type=str, default="data/silver/train.csv", help="Input CSV path with factual and satirical columns.")
    parser.add_argument("--output_dir", type=str, default="outputs/analysis", help="Directory for report and tables.")
    parser.add_argument("--top_k", type=int, default=15, help="How many top prefixes to report.")
    parser.add_argument("--jaccard_threshold", type=float, default=0.8, help="Threshold for high lexical overlap.")
    parser.add_argument("--seq_threshold", type=float, default=0.9, help="Threshold for high sequence similarity.")
    args = parser.parse_args()
    main(args)
