import os
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from selenium import webdriver
from pytube import YouTube
from moviepy.editor import *
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Define search term and language parameter
search_term = "Korean drama"
language_param = "kr"

# Launch Chrome browser with Selenium
driver = webdriver.Chrome()
driver.get(f"https://www.youtube.com/results?search_query={search_term}&sp={language_param}")

# Download top 10 videos
videos = []
for i in range(1, 11):
    video_url = driver.find_element_by_xpath(f'//*[@id="dismissable"]/div[{i}]/div[1]/div[1]/a/h3')
    video = YouTube(video_url.get_attribute("href"))
    videos.append(video)
for video in videos:
    video.streams.get_highest_resolution().download()

# Convert videos to WAV voice files
for video in videos:
    video_path = f"{video.title}.mp4"
    audio_path = f"{video.title}.wav"
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path)

# Ensemble pre-trained voice recognition models
gnb = GaussianNB()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
voting_clf = VotingClassifier(estimators=[("gnb", gnb), ("dtc", dtc), ("rfc", rfc)], voting="hard")
recognized_speeches = []
for video in videos:
    recognized_speech = ""
    with sr.AudioFile(f"{video.title}.wav") as source:
        audio = r.record(source)
        recognized_speech = r.recognize_google(audio, language=language_param)
    recognized_speeches.append(recognized_speech)
    print(recognized_speech)
ensemble_result = voting_clf.fit(recognized_speeches)

# Perform text data analysis with ensemble of sentiment analysis models
tokenizer_kolectra = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
model_kolectra = AutoModelForSequenceClassification.from_pretrained("monologg/koelectra-small-v3-discriminator")
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model_bert = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased")
tokenizer_korbert = AutoTokenizer.from_pretrained("snunlp/KR-BERT-char16424")
model_korbert = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-BERT-char16424")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_kolectra.to(device)
model_bert.to(device)
model_korbert.to(device)
input_ids_kolectra = tokenizer_kolectra.batch_encode_plus(recognized_speeches, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)["input_ids"]
input_ids_bert = tokenizer_bert.batch_encode_plus(recognized_speeches, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)["input_ids"]
input_ids_korbert = tokenizer_korbert.batch_encode_plus(recognized_speeches, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)["input_ids"]

# Perform text data analysis
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(recognized_speeches)
true_k = 2
model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100, n_init=1)
model.fit(X)
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print(f"Cluster {i+1}:")
    for ind in model.cluster_centers_.argsort()[i, :10]:
        print(f"{terms[ind]}")
lda_model = LatentDirichletAllocation(n_components=5, max_iter=10, learning_method="online", random_state=42)
lda_model.fit(X)
for i, topic in enumerate(lda_model.components_):
    print(f"Topic {i+1}:")
    print(" ".join([terms[j] for j in topic.argsort()[:-11:-1]]))
