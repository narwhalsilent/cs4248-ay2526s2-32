# Satirizing News Headlines: Unsupervised Style Transfer using Transformer Architectures

This is the codebase for CS4248 AY25/26 Semester 2 Project Group 32. We fine-tune transformer models to convert factual news headlines into satirical headlines while maintaining the original meaning and grammatical fluency. 

**Team Members:**
* Alicia Yap Zi Qi 
* Mahindroo Aashim 
* Wang Junwu 
* Anant Shanker 
* Sithanathan Rahul 
* Choong Kai Zhe 
* **Project Mentor:** Chen Xihao 

### Methodology

The pipeline is divided into distinct phases:

* **Phase 1: Synthetic Dataset Construction and Filtering** 
  * A teacher LLM. Llama 3, generates N=5 satirical rewrites for each factual headline.
  * Generation is guided by Pragmatic Metacognitive Prompting (PMP), which acts as a reasoning template to identify humor targets, violated norms, and ironic stances.
  * A rejection sampling pipeline filters candidates using a pre-trained sarcasm classifier for stylistic accuracy and BERTScore for semantic similarity.
  * The top-scoring candidate is retained to form the "Silver Dataset".
* **Phase 2: Model Architecture and Fine-Tuning** 
  * We utilize the encoder-decoder architectures of BART or T5.   
  * The models undergo Supervised Fine-Tuning (SFT) on the Silver Dataset using Cross-Entropy Loss.
* **Phase 3: Analysis and Interpretability** 
  * Ablation studies test different input formats, loss functions, and model sizes.
  * Interpretability is examined using SHAP to estimate word contributions.
  * Cross-attention maps are plotted to interpret token information flow.

### Evaluation Framework

The project employs a tri-factor evaluation framework alongside human validation:
* **Style Accuracy:** Measured by a frozen RoBERTa-based Sarcasm Detector (92% baseline).
* **Content Preservation:** Measured via Sentence-BERT (SBERT) cosine similarity.
* **Linguistic Fluency:** Measured using GPT-2 Perplexity.

## Installation

Install the required external libraries and frameworks:

```bash
pip install -r requirements.txt
```

If you use Anaconda, use
```bash
conda env create -f environment.yml
```

To use evaluate.py: 
```bash
python3 src/03_evaluation/evaluate.py \
  --model_path checkpoints/bart_satire/final \
  --test_data data/silver/test.csv \
  --num_examples 5
```

