import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
# nltk.download('stopwords')
# nltk.download('wordnet')

# Load your dataset
df = pd.read_csv('combined_news.csv')

# View columns and data info
print("Columns:", df.columns)
print("\nMissing Values:\n", df.isnull().sum())

# Drop rows where 'headline' is missing
df = df.dropna(subset=['headline'])

# Define text cleaning function
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)      # Keep only letters and spaces
    text = text.lower().strip()                  # Lowercase and trim
    return text

# Clean the headline text
df['cleaned_headline'] = df['headline'].apply(clean_text)

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Further process the text: remove stopwords and lemmatize
def preprocess_text(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['processed_headline'] = df['cleaned_headline'].apply(preprocess_text)

# Preview cleaned data
print(df[['headline', 'processed_headline']].head())

# Save the cleaned data
df.to_csv('cleaned_combined_news.csv', index=False)
print("\n✅ Cleaned data saved to 'cleaned_combined_news.csv'")
