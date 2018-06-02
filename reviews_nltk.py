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
df1 = df


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
cv = CountVectorizer()