#Social Media Sentiment Analysis

#Guide Line
import tweepy

# Set up authentication keys
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create API object
api = tweepy.API(auth)

# Define search query and specify language
query = 'your_topic'
language = 'en'

# Define list to store tweets
tweets = []

# Use Cursor to collect tweets
for tweet in tweepy.Cursor(api.search_tweets,
                           q=query,
                           lang=language,
                           tweet_mode='extended').items(1000):
    tweets.append(tweet.full_text)

# Print the first 5 tweets
print(tweets[:5])

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Define function to preprocess a single tweet
def preprocess_tweet(tweet):
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)
    
    # Remove hashtags and mentions
    tweet = re.sub(r'@\w+|#\w+', '', tweet)
    
    # Remove punctuation
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    
    # Convert to lowercase
    tweet = tweet.lower()
    
    # Tokenize tweet
    tokens = word_tokenize(tweet)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    # Join tokens back into string
    preprocessed_tweet = ' '.join(lemmatized_tokens)
    
    return preprocessed_tweet

# Define function to preprocess a list of tweets
def preprocess_tweets(tweets):
    preprocessed_tweets = []
    for tweet in tweets:
        preprocessed_tweet = preprocess_tweet(tweet)
        preprocessed_tweets.append(preprocessed_tweet)
    return preprocessed_tweets

# Example usage
tweets = ['This is a sample tweet with #hashtags and @mentions!',
          'Here is another tweet, with a link: https://www.example.com',
          'I love social media sentiment analysis!']
preprocessed_tweets = preprocess_tweets(tweets)
print(preprocessed_tweets)

from textblob import TextBlob

# Define function to annotate a single tweet with its polarity and subjectivity
def annotate_tweet(tweet):
    blob = TextBlob(tweet)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return {'tweet': tweet, 'polarity': polarity, 'subjectivity': subjectivity}

# Define function to annotate a list of tweets with their polarity and subjectivity
def annotate_tweets(tweets):
    annotated_tweets = []
    for tweet in tweets:
        annotated_tweet = annotate_tweet(tweet)
        annotated_tweets.append(annotated_tweet)
    return annotated_tweets

# Example usage
tweets = ['I love social media sentiment analysis!', 'This is a terrible day.']
annotated_tweets = annotate_tweets(tweets)
for tweet in annotated_tweets:
    print(tweet)

import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Define function to extract features from a list of annotated tweets
def extract_features(annotated_tweets):
    # Create a list of all words in the tweets
    words = [word for tweet in annotated_tweets for word in word_tokenize(tweet['tweet'])]
    # Calculate the frequency distribution of words
    fdist = FreqDist(words)
    # Extract the most common words and their frequencies
    top_words = fdist.most_common(10)
    features = {}
    # Add each word and its frequency to the features dictionary
    for word, freq in top_words:
        features[word] = freq
    # Add the average polarity and subjectivity of the tweets to the features dictionary
    polarity = sum([tweet['polarity'] for tweet in annotated_tweets])/len(annotated_tweets)
    subjectivity = sum([tweet['subjectivity'] for tweet in annotated_tweets])/len(annotated_tweets)
    features['polarity'] = polarity
    features['subjectivity'] = subjectivity
    return features

# Example usage
tweets = ['I love social media sentiment analysis!', 'This is a terrible day.']
annotated_tweets = annotate_tweets(tweets)
features = extract_features(annotated_tweets)
print(features)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the annotated data and extract features
annotated_tweets = load_annotated_data('tweets.csv')
features = extract_features(annotated_tweets)

# Convert the annotated data into a bag of words representation
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([tweet['tweet'] for tweet in annotated_tweets])
y = [tweet['label'] for tweet in annotated_tweets]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear support vector machine (SVM) model
model = LinearSVC()
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print(f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')

from sklearn.metrics import classification_report

# Load the annotated data and extract features
annotated_tweets = load_annotated_data('tweets.csv')
features = extract_features(annotated_tweets)

# Train a linear support vector machine (SVM) model
model = LinearSVC()
model.fit(X, y)

# Make predictions on a test set and print a classification report
test_tweets = load_test_data('test_tweets.csv')
X_test = vectorizer.transform([tweet['tweet'] for tweet in test_tweets])
y_test = [tweet['label'] for tweet in test_tweets]
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the annotated data and extract features
annotated_tweets = load_annotated_data('tweets.csv')
features = extract_features(annotated_tweets)

# Train a linear support vector machine (SVM) model
model = LinearSVC()
model.fit(X, y)

# Get the most informative features for each class
n = 50  # number of features to display
classes = model.classes_
coef = model.coef_
top_positive = sorted(zip(coef[0], features), reverse=True)[:n]
top_negative = sorted(zip(coef[1], features), reverse=True)[:n]

# Create a word cloud for positive and negative features
positive_text = ' '.join([f[1] for f in top_positive])
negative_text = ' '.join([f[1] for f in top_negative])
positive_wordcloud = WordCloud(background_color='white').generate(positive_text)
negative_wordcloud = WordCloud(background_color='white').generate(negative_text)

# Display the word clouds using matplotlib
plt.subplot(121)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Words')
plt.subplot(122)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Words')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the annotated data and extract features
annotated_tweets = load_annotated_data('tweets.csv')
features = extract_features(annotated_tweets)

# Train a linear support vector machine (SVM) model
model = LinearSVC()
model.fit(X, y)

# Make predictions on a test dataset
test_tweets = load_test_data('test_tweets.csv')
test_features = extract_features(test_tweets)
y_pred = model.predict(test_features)

# Load demographic information for the test dataset
demographics = pd.read_csv('test_demographics.csv')

# Compute sentiment distribution for each demographic group
sentiment_counts = demographics.groupby('gender')['sentiment'].value_counts(normalize=True)

# Plot the sentiment distribution for each demographic group
fig, ax = plt.subplots()
sentiment_counts.unstack().plot(kind='bar', stacked=True, ax=ax)
ax.set_xlabel('Gender')
ax.set_ylabel('Proportion of Tweets')
ax.set_title('Sentiment Distribution by Gender')
plt.show()
