# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:51:04 2019

@author: s-minhas
"""

import operator
import pandas as pd
import re
import sklearn
from sklearn.decomposition import PCA
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import ngrams
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.collocations import *
from nltk.corpus import webtext
import numpy as np
import random
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity  


def set_tokens_to_lowercase(data):
    for index, entry in enumerate(data):
        data[index] = entry.lower()
    
    return data


def remove_punctuation(data):
    symbols = ",.!"
    for index, entry in enumerate(symbols):
        for index2, entry2 in enumerate (data):
            data[index2] = re.sub(r'[^\w]', ' ', entry2)
    return data

def remove_stopwords_from_tokens(data):
       stop_words = set(stopwords.words("english"))
       new_list = []
       for index, entry in enumerate(data):
           no_stopwords = ""
           entry = entry.split()
           for word in entry:
               if word not in stop_words:
                    no_stopwords = no_stopwords + " " + word 
           new_list.append(no_stopwords)
       return new_list
   
    
def stemming (data):
    st = PorterStemmer()
    for index, entry in enumerate(data):
        data[index] = st.stem(entry)
    return data
  

raw_data_orig  = open("C:/IR Course/Adv -IR/IATI3.pkl","rb")
raw_data_orig = pickle.load(raw_data_orig, encoding='iso-8859-1')
raw_data_orig = raw_data_orig[raw_data_orig['description'].notnull()]

unprocessed_columns=list(raw_data_orig["iati-identifier"])
unprocessed = raw_data_orig['description']
unprocessed.index = unprocessed_columns


query ="prevent gender-based violence "

def preprocess(pdf):
    for index, row in pdf.iterrows():
           
            row['description'] = " ".join(stemming(remove_stopwords_from_tokens(remove_punctuation(set_tokens_to_lowercase(row['description'].split(" "))))))
    return pdf

#preprocess documents
raw_data= preprocess(raw_data_orig)


#now preprocess query
query = " ".join(stemming(remove_stopwords_from_tokens(remove_punctuation(set_tokens_to_lowercase(query.split(" "))))))
rownames = raw_data["iati-identifier"]


#vectorise and get tfidf values
vectorizer = TfidfVectorizer()
vectorized_iati = vectorizer.fit_transform(raw_data["description"])
tdm = pd.DataFrame(vectorized_iati.toarray(), columns = vectorizer.get_feature_names())
tdm=tdm.set_index(rownames)

#now vectorise query
vectorized_query=vectorizer.transform(pd.Series(query))
query = pd.DataFrame(vectorized_query.toarray(), columns = vectorizer.get_feature_names())

# get cosine similarity

def cos_sim (pdf, qdf):
    f_similarity={}   
    for index, row in qdf.iterrows():
        for index2, row2 in pdf.iterrows():
             cos_result = cosine_similarity(np.array(row).reshape(1, row.shape[0]), np.array(row2).reshape(1, row2.shape[0]))
             f_similarity[index2] = round(float(cos_result),5)
    return f_similarity

cosine_scores=cos_sim (tdm, query)
#now rank
final_rank= sorted(cosine_scores.items(), key=operator.itemgetter(1), reverse=True)

rownames = rownames.tolist()

for item in final_rank:
    if item[0] in rownames:
         
        print('IATI-IDENTIFIER {0} DESCRIPTION {1}'.format(item[0],unprocessed.iloc[rownames.index(item[0])])) 
        




