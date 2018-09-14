'''
Might be used:
We choose simple WORDNET to find synonyms and anyonyms by using NLTK
'''

from gensim import corpora, models, similarities
from util import read_xlsx_xlrd, read_csv, read_csv_tag
import logging
from pprint import pprint
import spacy
import string
from string import punctuation
import os
import re
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Build_Corpora(object):
    def __init__(self):
        self.documents = list()
        self.texts = list()

    def add(self, source):
        self.documents.append(str(source))
        return self.documents

    def word_filter(self, string):
        nlp = spacy.load('en')
        doc = nlp(string)
        result = list()
        for i, token in enumerate(doc):
            if token.pos_ in ('NOUN', 'PROPN', 'VERB', 'ADJ'):
                #print(doc[i])
                result.append(doc[i])
        #print(result)
        return result
    # remove common words and tokenize
    def clean(self):
        stop_list = set('for a of the and to in is with'.split())
        self.texts = [[word for word in self.documents.lower().split()]
                      for self.documents in self.documents]
        # remove words that appear only once
        #pprint(self.texts)
        return self.texts



    def remove_once_word(self):
        from collections import defaultdict
        frequency = defaultdict(int)
        for text in self.texts:
            for token in text:
                frequency[token] += 1
        self.texts = [[token for token in text if frequency[token] > 1]
                      for text in self.texts]
        return self.texts

    def dictionary(self, path):
        dictionary = corpora.Dictionary(self.texts)
        dictionary.save(path)
        #print(dictionary)
        #print(dictionary.token2id)
        return dictionary

    def corpus(self, path):
        corpus = [dictionary.doc2bow(text) for text in self.texts]
        corpora.MmCorpus.serialize(path, corpus)  # store to disk, for later use
       # print(corpus)
        return corpus

    def save_dictionary(self, path):
        pass


### Text Normalizing function. Part of the following function was taken from this link.
def clean_text(text):
    ## Remove puncuation
    text = text.translate(string.punctuation)

    ## Convert words to lower case and split them
    text = text.lower().split()

    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Stemming
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text

if __name__ == '__main__':
   # path = ('Data/Competency model 2 dimensional.xlsx')
    path = 'Data/Competency_model_2_dimensional.csv'
    bc_positive = Build_Corpora()
    # build positive competency model corpora

    # for i in range(29):
    #     #cell_positive = [i, 1, 2] # col: B
    #     content_cache = read_csv_tag(path, tag='Skilled')[i]
    #     #print(content_cache)
    #     documents_compe_positive = bc_positive.add(content_cache)
    # print(documents_compe_positive)



    """
    add word filter here.
    """
    for i in range(29):
        #cell_positive = [i, 1, 2] # col: B
        content_cache_1 = read_csv_tag(path, tag='Skilled')[i]
        content_cache_0 = read_csv_tag(path, tag='Competence')[i]
        content_cache = content_cache_0 + ' ' + content_cache_1
        print(content_cache)
        Positive_documents = clean_text(content_cache)
        print(Positive_documents)

        #documents_compe_positive = bc_positive.add(content_cache)
        documents_compe_positive = bc_positive.add(Positive_documents)
    #print(documents_compe_positive)

       # print(Positive_documents)
    #print(documents_compe_positive)





    # print(bc_negative.clean())
   # print(bc_positive.clean())
    bc_positive.clean()
    # bc_positive.save_dictionary('configure/positive.dict')
    #bc_positive.word_filter()
    # bc_negative.remove_once_word()
    # bc_negative.clean()
    dictionary = bc_positive.dictionary('configure/positive.dict')
    new_doc = clean_text('Focus on results')
    print(new_doc)
    vec_bow = dictionary.doc2bow(new_doc.lower().split())
    # print(new_vec)
    corpus = bc_positive.corpus('configure/positive.mm')
    # tfidf = models.TfidfModel(corpus)
    # corpus_tfidf = tfidf[corpus]
    '''Train the model using LDA'''
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
    corpus_lda = lda[corpus]
   # for doc in corpus_lda:
   #     print(doc)

    '''Send similarity queries'''

    #lda.print_topics(100)
    vec_lda = lda[vec_bow]
    index = similarities.MatrixSimilarity(corpus_lda)
    sims = index[vec_lda]

    print(list(enumerate(sims)))
    #print(documents_compe_negative)
