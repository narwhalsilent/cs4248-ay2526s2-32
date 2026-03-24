# Raw Dataset Quality Report: data/raw/Sarcasm_Headlines_Dataset_v2.json

## 1) Basic Stats
- Rows loaded: 28619
- Rows after cleaning: 28619
- Sarcastic headlines: 13634 (47.64%)
- Non-sarcastic headlines: 14985 (52.36%)

## 2) Source Domains
| Domain | Count | Percent |
|---|---:|---:|
| www.huffingtonpost.com | 14403 | 50.33% |
| www.theonion.com | 6577 | 22.98% |
| local.theonion.com | 3351 | 11.71% |
| politics.theonion.com | 2222 | 7.76% |
| entertainment.theonion.com | 1337 | 4.67% |
| www.huffingtonpost.comhttp: | 503 | 1.76% |
| sports.theonion.com | 123 | 0.43% |
| www.huffingtonpost.comhttps: | 79 | 0.28% |
| ogn.theonion.com | 24 | 0.08% |

## 3) All Headlines
- Rows: 28619
- Exact duplicate headlines: 116 (0.41%)
- Normalized duplicate headlines: 116 (0.41%)
- Avg headline length (tokens): 10.26
- Std headline length (tokens): 3.44
- Vocabulary size: 30807
- Type-token ratio: 0.1049
- Share of most common 2-word prefix: 0.66%
- Share covered by top 5 2-word prefixes: 2.42%

Top 2-word prefixes:

| Prefix | Count | Percent |
|---|---:|---:|
| area man | 189 | 0.66% |
| u s | 142 | 0.50% |
| how to | 141 | 0.49% |
| donald trump | 138 | 0.48% |
| hillary clinton | 83 | 0.29% |
| study finds | 77 | 0.27% |
| white house | 67 | 0.23% |
| this is | 66 | 0.23% |
| pope francis | 54 | 0.19% |
| man who | 46 | 0.16% |
| new study | 44 | 0.15% |
| bernie sanders | 42 | 0.15% |
| area woman | 39 | 0.14% |
| the best | 38 | 0.13% |
| supreme court | 35 | 0.12% |

Template marker frequencies:

| Marker | Count | Percent |
|---|---:|---:|
| nation horrified | 2 | 0.01% |
| new study reveals | 2 | 0.01% |
| local man | 30 | 0.10% |
| area man | 263 | 0.92% |
| experts baffled | 0 | 0.00% |
| white house | 213 | 0.74% |

## 4) Sarcastic Headlines Only
- Rows: 13634
- Exact duplicate headlines: 82 (0.60%)
- Normalized duplicate headlines: 82 (0.60%)
- Avg headline length (tokens): 10.59
- Std headline length (tokens): 3.89
- Vocabulary size: 20951
- Type-token ratio: 0.1450
- Share of most common 2-word prefix: 1.39%
- Share covered by top 5 2-word prefixes: 3.10%

Top 2-word prefixes:

| Prefix | Count | Percent |
|---|---:|---:|
| area man | 189 | 1.39% |
| u s | 77 | 0.56% |
| study finds | 72 | 0.53% |
| new study | 43 | 0.32% |
| white house | 42 | 0.31% |
| man who | 40 | 0.29% |
| area woman | 39 | 0.29% |
| pope francis | 38 | 0.28% |
| historical archives | 32 | 0.23% |
| hillary clinton | 25 | 0.18% |
| supreme court | 22 | 0.16% |
| high school | 20 | 0.15% |
| department of | 19 | 0.14% |
| man with | 19 | 0.14% |
| no one | 19 | 0.14% |

Template marker frequencies:

| Marker | Count | Percent |
|---|---:|---:|
| nation horrified | 2 | 0.01% |
| new study reveals | 2 | 0.01% |
| local man | 30 | 0.22% |
| area man | 263 | 1.93% |
| experts baffled | 0 | 0.00% |
| white house | 127 | 0.93% |

## 5) Non-Sarcastic Headlines Only
- Rows: 14985
- Exact duplicate headlines: 34 (0.23%)
- Normalized duplicate headlines: 34 (0.23%)
- Avg headline length (tokens): 9.97
- Std headline length (tokens): 2.93
- Vocabulary size: 20512
- Type-token ratio: 0.1374
- Share of most common 2-word prefix: 0.94%
- Share covered by top 5 2-word prefixes: 3.10%

Top 2-word prefixes:

| Prefix | Count | Percent |
|---|---:|---:|
| how to | 141 | 0.94% |
| donald trump | 135 | 0.90% |
| this is | 66 | 0.44% |
| u s | 65 | 0.43% |
| hillary clinton | 58 | 0.39% |
| the best | 38 | 0.25% |
| bernie sanders | 36 | 0.24% |
| donald trump's | 29 | 0.19% |
| here's what | 28 | 0.19% |
| how the | 28 | 0.19% |
| here's how | 27 | 0.18% |
| stephen colbert | 27 | 0.18% |
| ted cruz | 26 | 0.17% |
| white house | 25 | 0.17% |
| trevor noah | 25 | 0.17% |

Template marker frequencies:

| Marker | Count | Percent |
|---|---:|---:|
| nation horrified | 0 | 0.00% |
| new study reveals | 0 | 0.00% |
| local man | 0 | 0.00% |
| area man | 0 | 0.00% |
| experts baffled | 0 | 0.00% |
| white house | 86 | 0.57% |

## 6) Interpretation Guide
- High prefix concentration suggests repeated framing patterns.
- Duplicate rates reveal how much headline reuse exists in the raw data.
- Comparing sarcastic vs non-sarcastic subsets helps show whether one class is more templated or lexically narrow.