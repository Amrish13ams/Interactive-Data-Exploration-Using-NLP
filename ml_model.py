
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def load_data(file_path):
    return pd.read_csv(file_path)

def train_model(data):
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(data['review_text']).toarray()
    y = data['sentiment_label'].apply(lambda x: 1 if x == 'positive' else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    data = load_data('data/sentiment_data.csv')
    train_model(data)
    