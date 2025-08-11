# sentiment_visualization.py
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already present
nltk.download('stopwords', quiet=True)

def load_data():
    """Load the sentiment analysis results"""
    try:
        df = pd.read_csv('finbert_all_sentiment_scores.csv')
        print("✅ Data loaded successfully")
        return df
    except FileNotFoundError:
        print("❌ Error: File 'finbert_all_sentiment_scores.csv' not found. Please run Sentiment_news.py first.")
        return None

def plot_sentiment_distribution(df):
    """Plot pie chart of sentiment distribution"""
    plt.figure(figsize=(10, 8))
    sentiment_counts = df['predicted_sentiment'].value_counts()
    colors = {'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'gray'}
    explode = (0.05, 0.05, 0.05)  # Slightly explode all slices
    
    plt.pie(sentiment_counts, 
            labels=sentiment_counts.index, 
            autopct='%1.1f%%',
            colors=[colors[x] for x in sentiment_counts.index],
            explode=explode,
            startangle=90,
            shadow=True)
    
    plt.title('Distribution of Sentiment in Financial News Headlines', pad=20)
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_sentiment_scores(df):
    """Plot box plot of sentiment scores"""
    plt.figure(figsize=(12, 7))
    score_columns = ['positive_score', 'negative_score', 'neutral_score']
    box = df[score_columns].boxplot(patch_artist=True)
    
    # Customize colors
    colors = ['green', 'red', 'gray']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('Distribution of Sentiment Scores', pad=20)
    plt.ylabel('Sentiment Score')
    plt.xticks([1, 2, 3], ['Positive', 'Negative', 'Neutral'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('sentiment_scores_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_wordclouds(df):
    """Generate word clouds for each sentiment"""
    stop_words = set(stopwords.words('english'))
    
    def create_wordcloud(text, title, color):
        wordcloud = WordCloud(width=1200, height=600,
                              background_color='white',
                              stopwords=stop_words,
                              colormap=color).generate(text)
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud)
        plt.title(title, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'wordcloud_{title.split()[-1].lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Positive word cloud
    positive_text = ' '.join(df[df['predicted_sentiment'] == 'POSITIVE']['headline'])
    create_wordcloud(positive_text, 'Word Cloud for Positive Headlines', 'Greens')
    
    # Negative word cloud
    negative_text = ' '.join(df[df['predicted_sentiment'] == 'NEGATIVE']['headline'])
    create_wordcloud(negative_text, 'Word Cloud for Negative Headlines', 'Reds')
    
    # Neutral word cloud
    neutral_text = ' '.join(df[df['predicted_sentiment'] == 'NEUTRAL']['headline'])
    create_wordcloud(neutral_text, 'Word Cloud for Neutral Headlines', 'Greys')

def show_top_headlines(df):
    """Display top positive and negative headlines"""
    print("\nTOP 5 POSITIVE HEADLINES:")
    top_positive = df.nlargest(5, 'positive_score')[['headline', 'positive_score']]
    print(top_positive.to_string(index=False))
    
    print("\nTOP 5 NEGATIVE HEADLINES:")
    top_negative = df.nlargest(5, 'negative_score')[['headline', 'negative_score']]
    print(top_negative.to_string(index=False))

def plot_time_series(df):
    """Plot sentiment over time if date column exists"""
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Resample by week and get mean sentiment scores
            weekly_sentiment = df[['positive_score', 'negative_score', 'neutral_score']].resample('W').mean()
            
            plt.figure(figsize=(14, 7))
            weekly_sentiment.plot(linewidth=2.5)
            plt.title('Weekly Average Sentiment Scores Over Time', pad=20)
            plt.ylabel('Average Score')
            plt.xlabel('Date')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig('sentiment_over_time.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"⚠️ Could not create time series plot: {e}")

def main():
    """Main function to run all visualizations"""
    print("\n📊 Financial News Sentiment Analysis Visualizations\n")
    df = load_data()
    
    if df is not None:
        plot_sentiment_distribution(df)
        plot_sentiment_scores(df)
        generate_wordclouds(df)
        show_top_headlines(df)
        plot_time_series(df)
        print("\n🎉 All visualizations completed and saved as PNG files!")

if __name__ == "__main__":
    main()