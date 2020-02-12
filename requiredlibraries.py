
import pandas as pd
import numpy as np
import string
import operator
import re
import os
import sys
import codecs
import csv
import matplotlib.pylab as plt
from matplotlib.pyplot import figure
import wordcloud
from wordcloud import WordCloud
import nltk
from nltk import ngrams
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder 
from collections import Counter
from nltk.metrics import BigramAssocMeasures 
from nltk.collocations import *
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import gensim
from gensim.models import Word2Vec
from gensim import models, corpora
from gensim.models import TfidfModel
import sklearn
from sklearn.decomposition import PCA
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
plt.rcParams.update({'font.size': 7})
