
import jieba
import pandas as pd
import numpy as np
import csv
import logging
import json
import os
import pytest
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import gc


from keras.models import load_model

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] -%(levelname)s:%(message)s')
logging.info('information')
logging.debug('debug')
logging.error('something wrong')
logging.warning('warning')
logging.critical('critical wrong')
words = []

def vocab_w(path):
    with open(path, 'w', encoding='utf-8') as f:
        file = json.dumps(words)
        f.write(file)

def get_vocab(sentence):

    sen = jieba.cut(sentence,cut_all=True)
    for se in sen:
        if se not in words and se != ' ':
            words.append(se)


def cosine_sim(x, y):
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return cos



def get_one_code(sentence):
    bag =[]
    for word in words:
        bag.append(1) if word in [i for i in jieba.cut(sentence,cut_all=True)] else bag.append(0)
    return bag


def train_word2vec(documents,embedding_dim):
    model = Word2Vec(documents,min_count=1,size=embedding_dim)
    word_vectors = model.wv
    del model
    return word_vectors



if __name__ == '__main__':

    t = Tokenizer()
    p ='开始觉得活佛ID收费电视'

    print(t.fit_on_texts(p))





