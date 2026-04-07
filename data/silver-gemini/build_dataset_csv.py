import json
import pandas as pd
import os
from sklearn.model_selection import train_test_split

SATIRIZING_PATH = "scored_satirized_headlines.json"
DESATIRIZING_PATH = "scored_desatirized_headlines.json"
ALPHA = 0.6

def build_df_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Flatten the data structure
    rows = []
    for entry in data:
        original = entry.get('original_headline', '')
        candidates = entry.get('candidates', [])
        
        for cand in candidates:
            rows.append({
                'original_headline': original,
                'candidate_text': cand.get('text', ''),
                'content_score': cand.get('content_score', None),
                'style_score': cand.get('style_score', None)
            })

    df = pd.DataFrame(rows)

    df = df.dropna(subset=['content_score', 'style_score'])
    
    for col in ['content_score', 'style_score']:
        df[f'{col}_norm'] = (df[col] - df[col].mean()) / df[col].std()

    df['confidence_score'] = ALPHA * df['content_score_norm'] + (1 - ALPHA) * df['style_score_norm']
    
    df.drop(columns=['content_score_norm', 'style_score_norm', 'content_score'], inplace=True)

    winners = df.loc[df.groupby('original_headline')['confidence_score'].idxmax()]

    return winners

if __name__ == "__main__":
    satirizing_df = build_df_from_json(SATIRIZING_PATH)
    desatirizing_df = build_df_from_json(DESATIRIZING_PATH)

    combined_df = pd.concat([
        satirizing_df.rename(
            columns={'original_headline': 'factual', 'candidate_text': 'satirical'}
        ),
        desatirizing_df.rename(
            columns={'candidate_text': 'factual', 'original_headline': 'satirical'}
        )
    ], ignore_index=True)

    # Normalize confidence scores such that min is 0 and average is 1
    min_confidence = combined_df['confidence_score'].min()
    combined_df['confidence_score'] = combined_df['confidence_score'] - min_confidence
    avg_confidence = combined_df['confidence_score'].mean()
    combined_df['confidence_score'] = combined_df['confidence_score'] / avg_confidence

    if not os.path.exists('combined_data_full'):
        os.makedirs('combined_data_full')
    if not os.path.exists('combined_data_truncated'):
        os.makedirs('combined_data_truncated')

    train_df, temp_df = train_test_split(combined_df, test_size=0.2, random_state=42)
    test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    train_df_truncated = train_df.copy()
    train_df_truncated = train_df_truncated[train_df_truncated['style_score'] > 0.5]

    min_confidence_truncated = train_df_truncated['confidence_score'].min()
    train_df_truncated['confidence_score'] = train_df_truncated['confidence_score'] - min_confidence_truncated
    avg_confidence_truncated = train_df_truncated['confidence_score'].mean()
    train_df_truncated['confidence_score'] = train_df_truncated['confidence_score'] / avg_confidence_truncated

    train_df.drop(columns=['style_score'], inplace=True)
    test_df.drop(columns=['style_score'], inplace=True)
    val_df.drop(columns=['style_score'], inplace=True)
    train_df_truncated.drop(columns=['style_score'], inplace=True)

    train_df.to_csv('combined_data_full/train.csv', index=False)
    test_df.to_csv('combined_data_full/test.csv', index=False)
    val_df.to_csv('combined_data_full/val.csv', index=False)

    train_df_truncated.to_csv('combined_data_truncated/train.csv', index=False)
    test_df.to_csv('combined_data_truncated/test.csv', index=False)
    val_df.to_csv('combined_data_truncated/val.csv', index=False)