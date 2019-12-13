# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:07:22 2019

@author: s-minhas
"""

import nltk
import sys
import codecs
import nltk
from nltk.corpus import stopwords
import csv
import pandas
import re
import numpy as np


   
df = pandas.read_csv('C:/IR Course/Adv -IR/IATI10k.csv', header = 0, encoding="iso-8859-1")
df = df[df.description.notnull()]

def set_tokens_to_lowercase(data):
    for index, entry in enumerate(data):
        data[index] = entry.lower()
    
    return data

def remove_punctuation(data):
    symbols = ",.!"
    for index, entry in enumerate(symbols):
        for index2, entry2 in enumerate (data):
            data[index2] = re.sub(r'[^\w]', ' ', entry2)
            data[index2] = entry2.strip()
            
    return data

def remove_stopwords_from_tokens(data):
       stop_words = set(stopwords.words("english"))
       stop_words.add(" ")
       new_list = []
       for index, entry in enumerate(data):
               if entry not in stop_words:
                    new_list.append(entry)
       return new_list

def clean_df(pdf):
  
    for index, row in pdf.iterrows():
         row['description'] =  remove_stopwords_from_tokens(remove_punctuation(set_tokens_to_lowercase(row['description'].split())))  
         row['description'] = " ".join(x for x in row['description'])
    return pdf


def calc_docscore(pdf, pqry):
    col_names =  ['Description', 'score']
    f_df2  = pandas.DataFrame(columns = col_names)
    for index, row in pdf.iterrows(): 
        rank = []
        docscore = 0
        scored = score(row['description'])
        for word in pqry.split(" "):
            if word in scored.keys():
                rank.append(float(scored[word] )+ float(allcounts[word]/total)/2)
        
        if rank != []:
            docscore = np.prod(np.array(rank)) 
           
        f_df2.loc[index] = pandas.Series({'Description':row['description'], 'score':docscore})
    return f_df2
               
def score (pstr):
    fdict = {}
    flist = pstr.split()
    fdict = dict(nltk.FreqDist(flist))
    for  key, value in fdict.items():
        fdict[key] = round(fdict[key]/len(flist),2)
    return fdict

df = clean_df(df)
qry = "reduce transmission of HIV"
qry=  remove_stopwords_from_tokens(remove_punctuation(set_tokens_to_lowercase(qry.split())))
qry = " ".join(x for x in qry)      

allcounts = {} 
for descript in df['description']:
      tmp = dict(nltk.FreqDist(descript.split()))
      for key, value in tmp.items():
        if key not in allcounts:
            allcounts[key] = value
        else: 
            allcounts[key] = allcounts[key] + value
total = sum(allcounts.values())
df2=calc_docscore(df, qry) 
   
df2sort_by_score = df2.sort_values('score', ascending=False)
print (df2sort_by_score[1:20])
    
    
 