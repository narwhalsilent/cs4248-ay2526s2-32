# Satirizing News Headlines: Text Style Transfer (TST) using Transformer Architectures

This is the codebase for CS4248 AY25/26 Semester 2 Project Group 32. We conduct supervised fine-tuning (SFT) of transformer models to convert factual news headlines into satirical headlines while maintaining the original meaning and grammatical fluency. 

**Team Members:**
* Alicia Yap Zi Qi 
* Mahindroo Aashim 
* Wang Junwu 
* Anant Shanker 
* Sithanathan Rahul 
* Choong Kai Zhe 
* **Project Mentor:** Chen Xihao 

### Methodology

The base dataset used is News Headlines Dataset for Sarcasm Detection (NHDSD).

The pipeline is divided into distinct phases:

* **Phase 1: Synthetic Dataset Construction and Filtering** 
  * A teacher LLM. Gemma 3 12 B, generates N=5 satirical rewrites for each factual headline, and the same number of factual rewrites for each satirical headline.
  * Generation is guided by Pragmatic Metacognitive Prompting (PMP), which acts as a reasoning template to identify humor targets, violated norms, and ironic stances.
  * A rejection sampling pipeline filters candidates using English Sarcasm Detector for stylistic accuracy and BERTScore for semantic preservation.
  * The top-scoring candidate is retained to form the "Silver Dataset".
* **Phase 2: Model Architecture and Fine-Tuning** 
  * We utilize the encoder-decoder architectures of BART or T5.   
  * The models undergo Supervised Fine-Tuning (SFT) on the Silver Dataset using Cross-Entropy Loss.
* **Phase 3: Analysis and Evaluation** 
  * Ablation studies test different training datasets and training strategies.
  * Top performing models undergo additional Direct Policy Optimization (DPO) training.

### Evaluation Framework

The project employs a multi-factor evaluation framework:
* **Style Accuracy:** Measured by a frozen RoBERTa-based Sarcasm Detector (92% baseline).
* **Content Preservation:** Measured via Sentence-BERT (SBERT) cosine similarity.
* **Copy Rate:** Measured by proportion of N-grams shared between original and generated headlines.
* **Linguistic Fluency:** Measured using GPT-2 Perplexity.

In addtion, human evaluation is performed.

## Installation

Install the required external libraries and frameworks:

```bash
pip install -r requirements.txt
```

If you use Anaconda, use
```bash
conda env create -f environment.yml
```

