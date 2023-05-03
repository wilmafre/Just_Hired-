#IMPORTS
import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('wordnet')
import seaborn as sns
import numpy as np
import string
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

#Read dataset from 2022 csv-file

df = pd.read_csv("jobtech_dataset2022.csv")

#Remove/drop columns
columns_to_drop = ['removed_date', 'application_contacts', 'driving_license', 'access',
                    'salary_description', 'external_id', 'logo_url']
df.drop(columns_to_drop, inplace=True, axis=1)

# Convert the dataset to lowercase
def convert_to_lowercase(text):
    return text.lower()

# Remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

# Remove stopwords
def remove_stopwords(text):
    stopwords_list = STOPWORDS.union(set(['lärare'])) # Adding 'lärare' to the stopword list
    return ' '.join([word for word in simple_preprocess(text) if word not in stopwords_list])

# Lemmatize words
def lemmatize_stemming(text):
    stemmer = SnowballStemmer('swedish')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Preprocess text
def preprocess(text):
    text = convert_to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    result = []
    for token in simple_preprocess(text):
        if len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

# Tokenize and preprocess text data
documents = df['description'].map(preprocess)

# Create dictionary
dictionary = corpora.Dictionary(documents)

# Create corpus
corpus = [dictionary.doc2bow(document) for document in documents]

# Train LDA model
lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=5)

# Print topics
for idx, topic in lda_model.print_topics(num_topics=10):
    print('Topic: {} \nWords: {}'.format(idx, topic))
