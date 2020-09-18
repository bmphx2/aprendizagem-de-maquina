from autocorrect import Speller
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import sys
import glob
import os
import re
import heapq
import numpy as np
import pandas as pd
import sys
import numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
import pylab as pl
import seaborn as sns
import time
import json
import csv
from sklearn.svm import SVC

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

dataset = pd.read_csv('only_messages2.txt', encoding='utf-8')

data = []

dataset2 = pd.read_csv('test.txt', encoding='utf-8')

data2 = []

list_words = ['security', 'secure', 'vulnerable', 'leak', 'exception', 'crash', 'malicious',
              'sensitive', 'user', 'authentication', 'protect', 'vulnerability', 'authenticator', 'auth', 'npe']

for i in range(dataset.shape[0]):
    commits = dataset.iloc[i, 1]
    commits = re.sub('[^A-Za-z]', ' ', commits)
    commits = commits.lower()
    tokenized_commits = word_tokenize(commits)

    commits_processed = []
    for word in tokenized_commits:
        if word not in set(stopwords.words('english')):
            #spell = Speller(lang='en')
            # commits_processed.append(spell(stemmer.stem(word)))
            commits_processed.append(stemmer.stem(word))

    commits_text = " ".join(commits_processed)
    data.append(commits_text)

for i in range(dataset2.shape[0]):
    commits2 = dataset2.iloc[i, 1]
    commits2 = re.sub('[^A-Za-z]', ' ', commits2)
    commits2 = commits2.lower()
    tokenized_commits2 = word_tokenize(commits2)

    commits_processed2 = []
    for word in tokenized_commits2:
        if word not in set(stopwords.words('english')):
            #spell = Speller(lang='en')
            # commits_processed.append(spell(stemmer.stem(word)))
            commits_processed2.append(stemmer.stem(word))

    commits_text2 = " ".join(commits_processed2)
    data2.append(commits_text2)

# matrix = CountVectorizer()
# matrix.fit_transform(list_words)
# matrix.get_feature_names()
# print(matrix.transform(data).todense())

tf_counter = TfidfVectorizer(max_features=100, stop_words='english',
                             analyzer='word', use_idf=True, vocabulary=list_words)
X_train = tf_counter.fit_transform(data).toarray()

#X = matrix.fit_transform(data).toarray()
y_train = dataset.iloc[:, 0]
# print(X)

#X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.25, random_state = 10)
#X_train, y_train = train_test_split(X, y)

X_test = tf_counter.fit_transform(data2).toarray()
y_test = dataset2.iloc[:, 0]

#classifier = GaussianNB()
#classifier = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
#classifier = LinearDiscriminantAnalysis()
classifier = LogisticRegression(solver='lbfgs')
#classifier = Perceptron()
#classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
cr = classification_report(y_test, y_pred)
print(cr)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
f1s = str(f1_score(y_test, y_pred, average='weighted'))
print("F1 Score: ", f1s)


# print(tf_counter.get_feature_names())

#vectorize_maessage = matrix.transform(['this is a security issue vulnerability access']).toarray()
#vectorize_maessage = matrix.transform(['this is somegthing else']).toarray()
#vectorize_maessage = tf_counter.transform(['this is a security issue dude']).toarray()

#print (vectorize_maessage)
# print(classifier.predict(vectorize_maessage))
