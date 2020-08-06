from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import sys
import glob, os
import re,heapq
import numpy as np
import pandas as pd

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from autocorrect import spell

dataset = pd.read_csv('only_messages.txt', encoding='utf-8');

data = []

list_words = ['security', 'secure', 'issue', 'vulnerable', 'incorrect', 'access',
'failure', 'exception', 'overflow', 'null', 'ensure', 'leak', 'uninitialized',
'confusion', 'disclosure', 'use after free', 'user-after-free', 'malicious',
'confus', 'unauthor', 'un-author', 'auth', 'exploit', 'access', 'danger',
'bypass', 'sensitive', 'pass', 'safe', 'denial of service', 'cve', 'cwe', 'harmful','prevent','state']

for i in range(dataset.shape[0]):
    commits = dataset.iloc[i, 1]
    commits = re.sub('[^A-Za-z]', ' ', commits)
    commits = commits.lower()
    tokenized_commits = word_tokenize(commits)
 
    commits_processed = []
    for word in tokenized_commits:
        if word not in set(stopwords.words('english')):
        	#spell = Speller(lang='en')
        	commits_processed.append(spell(stemmer.stem(word)))

    commits_text = " ".join(commits_processed)
    data.append(commits_text)


matrix = CountVectorizer(max_features=200, ngram_range=(1,2))
matrix.fit_transform(list_words)
X = matrix.transform(data).toarray()
y = dataset.iloc[:, 0]
		

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.5, random_state = 5)
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
cr = classification_report(y_test, y_pred)
print(cr)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ",accuracy)
f1s = str(f1_score(y_test, y_pred, average='weighted'))
print("F1 Score: ",f1s)


