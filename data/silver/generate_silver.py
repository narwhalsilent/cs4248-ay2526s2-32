import json
import os
import time
import openai
import torch
import re
from datetime import datetime
from bert_score import scorer
from transformers import pipeline
from tqdm import tqdm

# --- 1. MODEL & RESOURCE SETUP ---
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Sarcasm Detector
sarcasm_pipe = pipeline(
    "text-classification", 
    model="helinivan/english-sarcasm-detector", 
    device=device,
    batch_size=5 
)

# BERTScore
bert_scorer = scorer.BERTScorer(
    lang="en", 
    rescale_with_baseline=True, 
    device=f"cuda:{device}" if device == 0 else "cpu"
)

# Ollama Client
client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def call_teacher_llm(headline):
    """
    Calls Ollama using JSON mode to ensure structured output.
    """
    pmp_system_prompt = (
        "You are an expert satirist for The Onion. Your task is to generate sarcasm.\n"
        "Process:\n"
        "1. Identify the TARGET. 2. Identify the SOCIAL NORM. 3. Identify the IRONIC STANCE.\n"
        "Generate exactly 5 distinct sarcastic headlines based on the provided factual headline.\n\n"
        "OUTPUT FORMAT: Respond ONLY with a JSON object. No prose before or after.\n"
        "Example: {\"headlines\": [\"Headline 1\", \"Headline 2\", \"Headline 3\", \"Headline 4\", \"Headline 5\"]}"
    )
    
    try:
        response = client.chat.completions.create(
            model="llama3:8b",
            messages=[
                {"role": "system", "content": pmp_system_prompt},
                {"role": "user", "content": f"Factual Headline: {headline}"}
            ],
            temperature=0.8,
            response_format={"type": "json_object"} # Forces valid JSON
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        headlines = data.get("headlines", [])
        
        # VALIDATION: Ensure we only return non-empty strings to prevent BERTScore crash
        clean_headlines = [str(h).strip() for h in headlines if h and str(h).strip()]
        
        return clean_headlines[:5]
    except Exception as e:
        print(f"\n[LLM ERROR]: {e}")
        return []

def generate_silver_dataset(input_path, output_path, log_path, limit=None):
    # --- 2. RESUME LOGIC ---
    processed_headlines = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_headlines.add(data['factual_headline'])
                except: continue
        print(f"Resuming: {len(processed_headlines)} headlines already processed.")

    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]

    to_process = [
        item for item in raw_data 
        if item.get('is_sarcastic') == 0 and item['headline'] not in processed_headlines
    ]
    
    if limit: to_process = to_process[:limit]
    if not to_process:
        print("No new headlines to process."); return

    print(f"Processing {len(to_process)} headlines. Logging to {log_path}...")

    # --- 3. MAIN LOOP ---
    with open(output_path, 'a', encoding='utf-8') as out_f, \
         open(log_path, 'a', encoding='utf-8') as log_f:
        
        log_f.write(f"\n--- New Session Started: {datetime.now()} ---\n")

        for idx, item in enumerate(tqdm(to_process), start=len(processed_headlines) + 1):
            start_time = time.time()
            original_text = item['headline']
            
            # Step 1: LLM Generation
            candidates = call_teacher_llm(original_text)
            
            # DEBUG PRINT
            print(f"\n[{idx}] Original: {original_text}")
            print(f"Candidates: {candidates}")

            # Safety check: If LLM failed or returned empty strings, skip this iteration
            if not candidates:
                print(f"Skipping index {idx} due to empty candidates.")
                continue

            # Step 2: Batched Scoring
            try:
                sarcasm_results = sarcasm_pipe(candidates)
                s_scores = [
                    res['score'] if res['label'] == 'LABEL_1' else 1 - res['score'] 
                    for res in sarcasm_results
                ]
                
                refs = [original_text] * len(candidates)
                _, _, f1_tensor = bert_scorer.score(candidates, refs)
                b_scores = f1_tensor.tolist()
            except Exception as e:
                print(f"Scoring Error at index {idx}: {e}")
                continue
            
            # Step 3: Winner Selection
            best_idx = 0
            max_total = -1
            candidate_logs = []

            for i in range(len(candidates)):
                total = (s_scores[i] + b_scores[i]) / 2
                candidate_logs.append({
                    "headline": candidates[i],
                    "sarcasm_score": round(s_scores[i], 4),
                    "bert_score": round(b_scores[i], 4),
                    "total_score": round(total, 4)
                })
                if total > max_total:
                    max_total, best_idx = total, i

            elapsed_time = time.time() - start_time

            # --- 4. LOGGING ---
            log_f.write(f"{idx}. {{'factual_headline': '{original_text}',\n")
            for i, c in enumerate(candidate_logs):
                marker = " [WINNER]" if i == best_idx else ""
                log_f.write(f"    'sarcastic_{i+1}': {{'headline': '{c['headline']}', 'score': {c['total_score']}}}{marker},\n")
            log_f.write(f"    'runtime': '{round(elapsed_time, 2)}s'}}\n\n")
            log_f.flush()

            # --- 5. SAVE SILVER PAIR ---
            result = {
                "factual_headline": original_text,
                "silver_sarcastic_headline": candidates[best_idx],
                "confidence_score": round(max_total, 4)
            }
            out_f.write(json.dumps(result) + '\n')
            out_f.flush()

if __name__ == "__main__":
    # Ensure these paths are correct relative to where you run the script
    INPUT_FILE = "../raw/Sarcasm_Headlines_Dataset.json"
    OUTPUT_FILE = "silver_headlines.jsonl"
    LOG_FILE = "generation_logs.txt"
    
    generate_silver_dataset(INPUT_FILE, OUTPUT_FILE, LOG_FILE, limit=100)