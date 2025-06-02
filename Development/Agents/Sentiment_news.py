import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Load cleaned news data
df = pd.read_csv('cleaned_combined_news.csv')

# Load FinBERT model
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create pipeline with return_all_scores=True
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)

# Function to extract all sentiment scores
def get_all_sentiment_scores(text):
    result = finbert(text)[0]
    scores = {entry['label']: entry['score'] for entry in result}
    return pd.Series([scores.get('POSITIVE', 0),
                      scores.get('NEGATIVE', 0),
                      scores.get('NEUTRAL', 0)])

# Apply function to get all scores
df[['positive_score', 'negative_score', 'neutral_score']] = df['processed_headline'].apply(get_all_sentiment_scores)

# Assign label based on max score
df['predicted_sentiment'] = df[['positive_score', 'negative_score', 'neutral_score']].idxmax(axis=1).str.replace('_score', '').str.upper()

# Preview
print(df[['headline', 'predicted_sentiment', 'positive_score', 'negative_score', 'neutral_score']].head())

# Save final CSV
df.to_csv('finbert_all_sentiment_scores.csv', index=False)
print("\n✅ All sentiment scores saved to 'finbert_all_sentiment_scores.csv'")
