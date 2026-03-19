import json
import os
import time
import openai
import torch
import warnings
from datetime import datetime
from bert_score import scorer
from transformers import pipeline
from tqdm import tqdm
import concurrent.futures

# Keep the terminal clean
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

# --- 1. MODEL & RESOURCE SETUP ---
device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device_name}")

# Sarcasm Detector - FP16 for TITAN V speed
sarcasm_pipe = pipeline(
    "text-classification", 
    model="helinivan/english-sarcasm-detector", 
    device=device_name,
    torch_dtype=torch.float16
)

# BERTScore - Standard model (No distillation)
# This will use the default robust model for maximum accuracy
bert_scorer = scorer.BERTScorer(
    lang="en", 
    rescale_with_baseline=True, 
    device=device_name
)

# Ollama Client
client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def call_teacher_llm(headline):
    pmp_system_prompt = (
        "You are an expert satirist for The Onion. Generate exactly 5 distinct sarcastic headlines.\n"
        "OUTPUT FORMAT: Respond ONLY with a JSON object. No prose.\n"
        "Example: {\"headlines\": [\"H1\", \"H2\", \"H3\", \"H4\", \"H5\"]}"
    )
    try:
        response = client.chat.completions.create(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": pmp_system_prompt},
                {"role": "user", "content": f"Factual Headline: {headline}"}
            ],
            temperature=0.8,
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        headlines = [str(h).strip() for h in data.get("headlines", []) if h and str(h).strip()]
        return headlines[:5]
    except Exception:
        return []

def process_batch(batch, sarcasm_pipe, bert_scorer, out_f):
    if not batch: return

    all_candidates = []
    item_maps = []
    for item in batch:
        all_candidates.extend(item['cands'])
        item_maps.append(len(item['cands']))

    with torch.inference_mode():
        # Batch Sarcasm Scores
        sarcasm_results = sarcasm_pipe(all_candidates, batch_size=len(all_candidates))
        s_scores = [
            res['score'] if res['label'] == 'LABEL_1' else 1 - res['score'] 
            for res in sarcasm_results
        ]

        # Batch BERTScore (Standard Model)
        all_refs = []
        for item in batch:
            all_refs.extend([item['original']] * len(item['cands']))
        
        _, _, f1_tensor = bert_scorer.score(all_candidates, all_refs)
        b_scores = f1_tensor.tolist()

    current_idx = 0
    for i, item in enumerate(batch):
        num_cands = item_maps[i]
        item_cands = item['cands']
        item_s_scores = s_scores[current_idx : current_idx + num_cands]
        item_b_scores = b_scores[current_idx : current_idx + num_cands]
        current_idx += num_cands

        scored_candidates = []
        for j in range(len(item_cands)):
            total = (item_s_scores[j] + item_b_scores[j]) / 2
            scored_candidates.append({"headline": item_cands[j], "total": total})
        
        scored_candidates.sort(key=lambda x: x['total'], reverse=True)
        winner = scored_candidates[0]

        tqdm.write(f"\n[ORIGINAL]: {item['original']}")
        tqdm.write(f" ★ WINNER: {winner['headline']} (Conf: {winner['total']:.3f})")

        result = {
            "factual_headline": item['original'],
            "silver_sarcastic_headline": winner['headline'],
            "confidence_score": round(winner['total'], 4)
        }
        out_f.write(json.dumps(result) + '\n')
        out_f.flush()

def generate_silver_dataset(input_path, output_path, limit=None):
    processed_headlines = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try: processed_headlines.add(json.loads(line)['factual_headline'])
                except: continue

    with open(input_path, 'r', encoding='utf-8') as f:
        to_process = [json.loads(line) for line in f if json.loads(line).get('is_sarcastic') == 0]
    
    to_process = [it for it in to_process if it['headline'] not in processed_headlines]
    if limit: to_process = to_process[:limit - len(processed_headlines)]
    
    print(f"Resuming: {len(processed_headlines)} headlines done. Starting {len(to_process)}...")

    batch_buffer = []
    # Batch size lowered to 6 because RoBERTa-large is VRAM-heavy
    batch_size = 6 

    with open(output_path, 'a', encoding='utf-8') as out_f:
        # Keep 8 workers for LLM generation
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_head = {executor.submit(call_teacher_llm, it['headline']): it['headline'] for it in to_process}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_head), total=len(to_process)):
                orig_text = future_to_head[future]
                try:
                    candidates = future.result()
                    if candidates:
                        batch_buffer.append({"original": orig_text, "cands": candidates, "start": time.time()})
                    
                    if len(batch_buffer) >= batch_size:
                        process_batch(batch_buffer, sarcasm_pipe, bert_scorer, out_f)
                        batch_buffer = []
                except Exception as e:
                    tqdm.write(f"Error: {e}")

            if batch_buffer:
                process_batch(batch_buffer, sarcasm_pipe, bert_scorer, out_f)

if __name__ == "__main__":
    INPUT_FILE = "../raw/Sarcasm_Headlines_Dataset.json"
    OUTPUT_FILE = "silver_headlines.jsonl"
    generate_silver_dataset(INPUT_FILE, OUTPUT_FILE, limit=None)
