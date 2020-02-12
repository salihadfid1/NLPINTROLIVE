# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:14:08 2019

@author: s-minhas
"""

import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk import FreqDist
import operator
import matplotlib.pylab as plt
from matplotlib.pyplot import figure
from wordcloud import WordCloud
from nltk.collocations import BigramCollocationFinder 
from nltk.metrics import BigramAssocMeasures 
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.rcParams.update({'font.size': 7})
import operator
from sklearn.feature_extraction.text import TfidfVectorizer

#wd = r'C:/IR Course/NLP_Intro_Course-master/spamham'


def split_text_to_tokens(text):
    return nltk.word_tokenize(text)

def remove_punctuation_from_tokens(tokens):
    #punctuation = ['(', ')', '?', ':', ';', ',', '.', '!', '/', '"', "'"] 
    punctuation='!?,.:;"\')(_-'
    text_without_punctuations = []
    for entity in tokens:
        newstring = ""
        for char in entity:
            if(char not in punctuation):
                  newstring+= char
        text_without_punctuations.append(newstring)
    return text_without_punctuations
  

def remove_non_alphabetic_tokens(tokens):
    alphabetic_tokens = []
    for token in tokens:
        if token.isalpha():
            alphabetic_tokens.append(token)
    return alphabetic_tokens



def remove_stopwords_from_tokens(tokens):
    stop_words = set(stopwords.words("english"))
    return [each_token for each_token in tokens if each_token not in stop_words]


def set_tokens_to_lowercase(tokens):
    
    return [each_token.lower() for each_token in tokens]

def preprocess(pstr1):
    
     s=split_text_to_tokens(pstr1)
     s=remove_non_alphabetic_tokens(s)
     s=remove_punctuation_from_tokens(s)
     s=set_tokens_to_lowercase(s)
     return s
 
##################

raw_data = pd.read_csv("C:/IR Course/NLP_Intro/SMSSpamCollection.csv",  encoding='iso-8859-1') 

raw_data["Email"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = True)
plt.ylabel("Spam vs Ham")
plt.legend(["Ham", "Spam"])
plt.show()


def getsampledata(pdf, psamp):
    types = ['spam', 'ham']
    allsamples = pd.DataFrame()
    for i in types:
        data1 = pdf[pdf.Email == i]
        rows = np.random.choice(data1.index.values, psamp)
        sampled_data = pdf.loc[rows] 
        allsamples = allsamples.append(sampled_data, ignore_index=True)

    return allsamples

samp_data = getsampledata (raw_data, 10)

def populatedictcorpus(data):
    pdict1 = {} 
    textspam = ""
    textham = ""
    list_WtV_spam=[]
    list_WtV_ham=[]
    
    for index, row in data.iterrows():
            if row['Email']=='spam':
                                      #s= preprocess(row['Description'])
                                      #textspam = textspam + " ".join(s) 
                                      textspam = row['Description'] + " " +  textspam
                                      list_WtV_spam.append(row['Description'].split(" ")) 
 
            else:   
                                      #s= preprocess(row['Description'])
                                      #textham = textham + " ".join(s) 
                                      textham = row['Description'] + " " +  textham
                                      list_WtV_ham.append(row['Description'].split(" ")) 
                                      
                                      
   
    pdict1.update({'spam': textspam})
    pdict1.update({'ham': textham})
   
    alldata = [pdict1,list_WtV_spam, list_WtV_ham ]
   
    return alldata
   
  
def freqalltokens(palltext):
        
            dictcounts = {}
            palltext = palltext.split(" ")
            for token in palltext:
                if token in dictcounts:
                    dictcounts[token] = dictcounts[token] + 1
                else:
                     dictcounts[token] =  1
            sorted_val = sorted(dictcounts.items(), key=operator.itemgetter(1), reverse=True)         
            return sorted_val          


def plotall(px, py):

    plt.xticks(fontsize=6, rotation=90)
    plt.ylabel('Frequency')
    plt.plot(px, py)
    plt.show()



def percentoftotal (px, py, psum):
    pfreqdict = {}
    for i in range (10):
         pfreqlist = []
         t= py[i]/psum *100
         pfreqlist.append(py[i])
         pfreqlist.append(round(t, 3))
         t = py[i]/psum * 1000
         pfreqlist.append(round(t,3))
         pfreqdict[px[i]] = pfreqlist
    return pfreqdict

####lexical diversity

def lexical_diversity(text):
  
    info = []
    info.append(len(text))
    info.append(len(set(text))) 
    info.append(len(set(text))/len(text))
    return info

         
count_spamham = []
sum_tokens=0
alltext = ""
complete_list = populatedictcorpus(samp_data)
dict1 = complete_list[0]
for key in dict1:
    count_spamham.append([key, len(dict1[key])])
    sum_tokens=len(dict1[key]) + sum_tokens
    alltext = alltext + dict1[key]

a=freqalltokens(alltext)

#x, y = zip(*freqalltokens(alltext))

token = []
count = []
for item in a:
    token.append(item[0])
    count.append(item[1])
    
 
import csv

with open("C:/IR Course/NLP_Intro/datafreq.csv", mode='w', newline='', encoding='iso-8859-1') as datafreq:
    datafreq = csv.writer(datafreq, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    datafreq.writerow(token)
    datafreq.writerow(count)

 
    
    
plotall(token, count)


for itemnum in range (len(count_spamham)):
    print ("Number of tokens in:", count_spamham[itemnum][0], count_spamham[itemnum][1])
print ("Number of tokens in text:", sum_tokens)



freqdict= percentoftotal (token, count, sum_tokens)

##plot
tokens = tuple(freqdict.keys())
values = freqdict.values()
total, percent,normalised = zip(*values)


plotall(tokens, normalised)


lingstats = lexical_diversity(dict1['spam'] + " " + dict1['ham'])
print ("Total tokens:", lingstats[0], "Total Unique Words:", lingstats[1], "Type/Token Ratio:", round(lingstats[2], 6))


#Spam Word cloud

def words_to_cloud (pstr):
    wordcloud = WordCloud().generate(pstr)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    
words_to_cloud (" ".join(dict1['spam'].split(" ")))

#dispersion
#spam_text_tokens = nltk.word_tokenize(dict1['spam']) #tokenize
spam_text_tokens = nltk.word_tokenize(dict1['spam']) #tokenize
spam_text_object = nltk.Text(spam_text_tokens) #turning it into nltk.Text object to be able to use .condordance, .similar etc
spam_text_object.dispersion_plot(["call", "service", "text"])

###################
   
# get concordance

allspamtokens = nltk.word_tokenize(dict1['spam']) #tokenize
spamtoken_object = nltk.Text(allspamtokens) #turning it into nltk.Text object to be able to use .condordance, .similar etc
spamtoken_object.concordance('call')

################

##Get bigrams

#bigrm = nltk.bigrams(dict1['ham'].split(" "))
#biagram_collocation = BigramCollocationFinder.from_words(dict1['ham'].split(" ")) 
#biagram_collocation.nbest(BigramAssocMeasures.likelihood_ratio, 15) 


def generate_collocations(tokens):
    '''
    Given list of tokens, return collocations.
    '''

    ignored_words = nltk.corpus.stopwords.words('english')
    bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(tokens)
    bigramFinder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
    bigram_freq = bigramFinder.ngram_fd.items()
    bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram','freq']).sort_values(by='freq', ascending=False)
   
    return bigramFreqTable

print (generate_collocations(dict1['spam'].split()))





###plot embeddings
complete_list = populatedictcorpus(raw_data)
#spam
model = Word2Vec(complete_list[1], min_count=20,size=50,workers=4)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['win'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)

#plot with PCA

#ham

model2 = Word2Vec(complete_list[2], min_count=20,size=50,workers=4)
# summarize the loaded model
print(model2)
# summarize vocabulary
words2 = list(model2.wv.vocab)
print(words2)
# access vector for one word
print(model2['guy'])
# save model
model2.save('model2.bin')
# load model
new_model2 = Word2Vec.load('model2.bin')
print(new_model2)

# dimensionality reduction 
X = model[model.wv.vocab]
X2 = model2[model2.wv.vocab]


pca1 = PCA(n_components=2)
result = pca1.fit_transform(X)

pca2 = PCA(n_components=2)
result2 = pca2.fit_transform(X2)

# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(result[:, 0], result[:, 1], c="red",s=5,label="spam")
ax.scatter(result2[:, 0], result2[:, 1], c="blue",s=5,label="ham")
plt.xlim(-0.50, 1.25) 
plt.ylim(-0.04, 0.04)
plt.gcf().set_size_inches((10, 10))   


words = list(model.wv.vocab)
for i, word in enumerate(words):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))


words2 = list(model2.wv.vocab)
for i, word2 in enumerate(words2):
	plt.annotate(word2, xy=(result2[i, 0], result2[i, 1]))


plt.title('Spam Ham Embeddings')
plt.legend(loc=2)


plt.show()

##separate

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4), sharey=True, dpi=120)

# Plot
ax1.scatter(result[:, 0], result[:, 1], c="red",label="spam", s= 5)

ax2.scatter(result2[:, 0], result2[:, 1], c="blue",label="ham", s= 5)

# Title, X and Y labels, X and Y Lim
ax1.set_title('Spam Embeddings'); ax2.set_title('Ham Embeddings')
ax1.set_xlabel('X');  ax2.set_xlabel('X')  # x label
ax1.set_ylabel('Y');  ax2.set_ylabel('Y')  # y label
ax1.set_xlim(-0.50, 1.25) ;  ax2.set_xlim(-0.50, 1.25)   # x axis limits
ax1.set_ylim(-0.04, 0.04);  ax2.set_ylim(-0.04, 0.04)  # y axis limits


words = list(model.wv.vocab)
for i, word in enumerate(words):
	ax1.annotate(word, xy=(result[i, 0], result[i, 1]), fontsize=5)


words2 = list(model2.wv.vocab)
for i, word2 in enumerate(words2):
	ax2.annotate(word2, xy=(result2[i, 0], result2[i, 1]), fontsize=5)


ax1.legend(loc=2)
ax2.legend(loc=5)


# ax2.yaxis.set_ticks_position('none') 
plt.tight_layout()
plt.show()


      






                          
          

















