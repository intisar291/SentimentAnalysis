import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import re
import string
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download("stopwords")
from gensim.test.utils import common_dictionary, common_corpus
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB,ComplementNB
from sklearn.metrics import classification_report
import itertools
from wordcloud import WordCloud,ImageColorGenerator,STOPWORDS
import matplotlib.pyplot as plt

# Initiating the Sentiment() class by providing file path of the labeled dataset
new_analysis = Sentiment(r'/path/myComputer/fileLocation/data.xlsx') 

# It is possible to change the parameters of all this methods except .label_encoder()

new_analysis.label_encoder()
new_analysis.pre_processing()
new_analysis.svm()
new_analysis.multi_NB()
new_analysis.comp_NB()
