
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

def load_data(file_path):
    return pd.read_csv(file_path)

def sentiment_analysis(data):
    sia = SentimentIntensityAnalyzer()
    data['sentiment'] = data['review_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    data['sentiment_label'] = data['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')
    return data

if __name__ == "__main__":
    data = load_data('data/cleaned_data.csv')
    sentiment_data = sentiment_analysis(data)
    sentiment_data.to_csv('data/sentiment_data.csv', index=False)
    print("Sentiment analysis completed and saved.")
    