# Silver Dataset Quality Report: data/silver/train.csv

## 1) Basic Stats
- Rows loaded: 11957
- Rows after cleaning: 11957
- Exact duplicate satirical headlines: 4 (0.03%)
- Normalized duplicate satirical headlines: 6 (0.05%)

## 2) Template Concentration
- Share of most common 2-word prefix: 9.96%
- Share covered by top 5 2-word prefixes: 24.92%

Top 2-word prefixes:

| Prefix | Count | Percent |
|---|---:|---:|
| nation horrified | 1191 | 9.96% |
| local man | 599 | 5.01% |
| new study | 593 | 4.96% |
| area man | 321 | 2.68% |
| local man's | 276 | 2.31% |
| area man's | 142 | 1.19% |
| local woman | 123 | 1.03% |
| local woman's | 77 | 0.64% |
| experts baffled | 67 | 0.56% |
| area woman | 49 | 0.41% |
| study finds | 43 | 0.36% |
| area residents | 37 | 0.31% |
| white house | 35 | 0.29% |
| new poll | 34 | 0.28% |
| study reveals | 31 | 0.26% |

Template marker frequencies:

| Marker | Count | Percent |
|---|---:|---:|
| nation horrified | 1212 | 10.14% |
| new study reveals | 374 | 3.13% |
| local man | 892 | 7.46% |
| area man | 465 | 3.89% |
| experts baffled | 94 | 0.79% |
| white house | 69 | 0.58% |

## 3) Diversity and Length
- Avg satirical length (tokens): 13.18
- Std satirical length (tokens): 3.39
- Vocabulary size (satirical): 19587
- Type-token ratio (satirical): 0.1243

## 4) Source-Target Overlap
- Mean lexical Jaccard overlap: 0.1228
- Median lexical Jaccard overlap: 0.1000
- Mean normalized sequence similarity: 0.3662
- Median normalized sequence similarity: 0.3562
- Exact copies (source == target after normalization): 3 (0.03%)
- High lexical overlap (Jaccard >= 0.80): 5 (0.04%)
- High sequence similarity (ratio >= 0.90): 7 (0.06%)

## 5) Interpretation Guide
- High prefix concentration suggests style template collapse.
- High source-target overlap suggests weak rewrites (too close to factual headlines).
- Low diversity metrics suggest limited stylistic coverage for training.

## 6) Artifacts
- 2-word prefix table: outputs/analysis/train_top_prefix2.csv
- 3-word prefix table: outputs/analysis/train_top_prefix3.csv