import json

input_file = 'processed_headlines_desatirized.json'
output_file = 'prettified_desatirized_headlines.json'

data = []

# Read the JSON Lines file
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

# Write as a single prettified JSON array
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Done! Prettified file saved as {output_file}")