import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Step 1: Load Cleaned News Data
df = pd.read_csv('cleaned_combined_news.csv')

# Step 2: Load FinBERT Model and Tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Step 3: Set Up Sentiment Analysis Pipeline
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)

# Step 4: Define Function to Extract Sentiment Scores
def get_all_sentiment_scores(text):
    result = finbert(text)[0]
    scores = {entry['label'].upper(): entry['score'] for entry in result}
    return pd.Series([  
        scores.get('POSITIVE', 0),
        scores.get('NEGATIVE', 0),
        scores.get('NEUTRAL', 0)
    ])

# Step 5: Apply to Headlines Column
df[['positive_score', 'negative_score', 'neutral_score']] = df['headline'].apply(get_all_sentiment_scores)

# Step 6: Assign Final Sentiment Label Based on Max Score
df['predicted_sentiment'] = df[['positive_score', 'negative_score', 'neutral_score']]\
    .idxmax(axis=1).str.replace('_score', '').str.upper()

# Step 7: Preview Result
print(df[['headline', 'predicted_sentiment', 'positive_score', 'negative_score', 'neutral_score']].head())

# Step 8: Save to New CSV
df.to_csv('finbert_all_sentiment_scores.csv', index=False)
print("\n✅ All sentiment scores saved to 'finbert_all_sentiment_scores.csv'")
