# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:43:53 2018

@author: trust-tyler
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the dataset
df = pd.read_csv("Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)
df


#Cleaning the texts
#Removing words like the and punctation like "..." as these thing have no meaning
#also applying stemming taking the root of the word(e.g loved is the past tense of love so love is selected)

#^specifies what we don't want to remove, second parameter means that the removed characters will be replaced by a space
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
review = re.sub('[^a-zA-Z]', ' ', df["Review"][0])
#convert to small case
review = review.lower()
review = review.split()
#remove stop words from review which holds no meaning to help our model
#'english' is given as param as there are multiple stopwords in diff languages
#set is used so that it can execute faster in case of big sentences
ps = PorterStemmer()
review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
review = ' '.join(review)


#cleaning the text for every review

corpus = []
for i in range(len(df)):
    review1 = re.sub('[^a-zA-Z]', ' ', df["Review"][i])
    review1 = (review1.lower()).split()
    ps = PorterStemmer()
    review1 = [ps.stem(word1) for word1 in review1 if word1 not in set(stopwords.words('english'))]
    review1 = ' '.join(review1)
    corpus.append(review1)
    
#Creating bag of words model
    
#in this model we will have 1000 rows for 1000 reviews and there will be a column for each word and the cell
#coresponding to that word will contain count if that word is present in that review or not
#since here we will have multiple 0 we will have a sparse matrix

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) 
#as currently there too many columns and it also contains columns for words that are not too frequent
#like names of people or even words which only occur one or two times
x = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values

#final classification through naive bayes 
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

#accuracy of the model

(cm[0][0]+cm[1][1])/200