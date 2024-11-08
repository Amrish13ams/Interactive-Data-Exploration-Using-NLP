
import pandas as pd
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(data):
    data = data.drop_duplicates()
    data = data.dropna(subset=['review_text'])
    return data

if __name__ == "__main__":
    data = load_data('data/raw_data.csv')
    cleaned_data = clean_data(data)
    cleaned_data.to_csv('data/cleaned_data.csv', index=False)
    print("Data cleaned and saved.")
    