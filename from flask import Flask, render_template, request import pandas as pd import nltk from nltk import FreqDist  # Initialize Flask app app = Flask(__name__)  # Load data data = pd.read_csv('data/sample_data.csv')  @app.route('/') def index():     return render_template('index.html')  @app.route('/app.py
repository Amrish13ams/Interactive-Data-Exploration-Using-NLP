from flask import Flask, render_template, request
import pandas as pd
import nltk
from nltk import FreqDist

# Initialize Flask app
app = Flask(__name__)

# Load data
data = pd.read_csv('data/sample_data.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/explore', methods=['POST'])
def explore():
    text = request.form['text']
    tokens = nltk.word_tokenize(text)
    freq_dist = FreqDist(tokens)
    most_common = freq_dist.most_common(10)
    return render_template('results.html', results=most_common)

if __name__ == '__main__':
    app.run(debug=True)
