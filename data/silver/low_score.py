import json

def count_low_confidence_scores(file_path, threshold=0.2):
    low_confidence_count = 0
    total_processed = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    # Extract the score, defaulting to 1.0 if missing to avoid false positives
                    score = data.get('confidence_score', 1.0)
                    
                    if score < threshold:
                        low_confidence_count += 1
                    
                    total_processed += 1
                except json.JSONDecodeError:
                    print(f"Skipping malformed JSON line: {line[:50]}...")
                    continue

        print(f"Processing Complete.")
        print(f"Total lines checked: {total_processed}")
        print(f"Lines with confidence score < {threshold}: {low_confidence_count}")
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")

if __name__ == "__main__":
    # Ensure the path matches your directory structure
    FILE_NAME = "silver_headlines.jsonl"
    count_low_confidence_scores(FILE_NAME)