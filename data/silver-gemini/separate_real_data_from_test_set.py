import json
import pandas as pd

DESATIRIZING_PATH = "scored_desatirized_headlines.json"
TEST_CSV_PATH = "combined_data_full/test.csv"
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
    desatirizing_df = build_df_from_json(DESATIRIZING_PATH)
    desatirizing_df.rename(columns={'original_headline': 'satirical', 'candidate_text': 'factual'}, inplace=True)
    desatirizing_df_good = desatirizing_df[desatirizing_df['style_score'] > 0.5]
    desatirizing_df.drop(columns=['style_score', 'confidence_score'], inplace=True)
    desatirizing_df_good.drop(columns=['style_score', 'confidence_score'], inplace=True)

    test_df = pd.read_csv(TEST_CSV_PATH)

    test_synthetic = pd.merge(test_df, desatirizing_df, on=['satirical', 'factual'], how='inner')
    test_synthetic_good = pd.merge(test_df, desatirizing_df_good, on=['satirical', 'factual'], how='inner')
    test_real = test_df[~test_df.index.isin(test_synthetic.index)]
    

    test_real.to_csv("combined_data_full/test_real.csv", index=False)
    test_synthetic.to_csv("combined_data_full/test_synthetic.csv", index=False)
    test_synthetic_good.to_csv("combined_data_full/test_synthetic_good.csv", index=False)