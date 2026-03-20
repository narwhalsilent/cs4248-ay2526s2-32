"""
prepare_data.py

Converts the generated silver_headlines.jsonl into train/val/test CSVs
that are consumed by train.py. Run this once before training.

Usage:
    python src/01_data_generation/prepare_data.py
    python src/01_data_generation/prepare_data.py --min_confidence 0.35
"""

import argparse
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main(args):
    input_path = args.input
    output_dir = args.output_dir

    # 1. Load the JSONL silver dataset
    print(f"Loading silver dataset from: {input_path}")
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    print(f"Total pairs loaded: {len(df)}")

    # 2. Rename columns to match what train.py expects
    df = df.rename(columns={
        "factual_headline": "factual",
        "silver_sarcastic_headline": "satirical"
    })

    # 3. Drop rows with missing values
    before = len(df)
    df = df.dropna(subset=["factual", "satirical"])
    df = df[df["factual"].str.strip() != ""]
    df = df[df["satirical"].str.strip() != ""]
    print(f"Dropped {before - len(df)} rows with empty values. Remaining: {len(df)}")

    # 4. Optional: filter by confidence score
    if args.min_confidence > 0:
        before = len(df)
        df = df[df["confidence_score"] >= args.min_confidence]
        print(f"Filtered to confidence >= {args.min_confidence}. Dropped {before - len(df)} low-quality pairs. Remaining: {len(df)}")

    # 5. Keep only the columns train.py needs
    df = df[["factual", "satirical"]]

    # 6. Split into train / val / test (80 / 10 / 10)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print("\nSplit sizes:")
    print(f"  Train : {len(train_df)}")
    print(f"  Val   : {len(val_df)}")
    print(f"  Test  : {len(test_df)}")

    # 7. Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"\nSaved CSVs to: {output_dir}")
    print("Done. Ready for training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare silver dataset for training.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/silver/silver_headlines.jsonl",
        help="Path to the silver_headlines.jsonl file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/silver",
        help="Directory to save train.csv, val.csv, test.csv."
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.0,
        help="Minimum confidence score to include a pair (e.g. 0.35 to filter weak pairs)."
    )
    args = parser.parse_args()
    main(args)
