from flask import Flask, render_template, request

import nltk
import re
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# Download necessary NLTK resources
nltk.download('vader_lexicon')

# Initialize the VADER sentiment intensity analyzer
preTrainedModel = SentimentIntensityAnalyzer()

def clean_text(text):
    tweet_words = []
    for word in text.lower().split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    return tweet_words

def perform_sentiment_analysis(tweet):
    # Clean the tweet
    cleaned_tweet = ' '.join(clean_text(re.sub(r'^RT[\s]+', '', tweet)))

    # Perform sentiment analysis
    sentiment_scores = preTrainedModel.polarity_scores(cleaned_tweet)

    # Determine sentiment category
    if sentiment_scores['compound'] >= 0.05:
        sentiment = 'positive'
    elif -0.05 <= sentiment_scores['compound'] < 0.05:
        sentiment = 'neutral'
    else:
        sentiment = 'negative'

    return sentiment, sentiment_scores
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        tweet_to_analyze = request.form['tweet']
        result_sentiment, result_scores = perform_sentiment_analysis(tweet_to_analyze)
        return render_template('result.html', tweet=tweet_to_analyze, sentiment=result_sentiment, scores=result_scores)

if __name__ == '__main__':
    app.run(debug=True)
