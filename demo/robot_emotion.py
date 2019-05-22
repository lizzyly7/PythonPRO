import os
import jieba
import re
import numpy as np
import random
import csv
import zhon
from zhon.hanzi import punctuation
import json
import string
import logging

path = os.path.dirname(__file__)
data_path = r'C:\Users\Administrator\pyhtonPRO\test\emotion'
classes = {}
words = []
label = []
label_type = []
training_Data = []
dir_dict = {}
dict_dir = {}
w2v_path = r'C:\Users\Administrator\Desktop\PythonPRO\demo\w2v.model'



def get_words(path):
    word = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line and len(line) > 1 and line not in word:
                word.append(line.strip())
        return word


def get_data(path=data_path):
    if os.path.exists(path):
        i = 0
        for file in os.listdir(path):
            dir_dict[file[:-4]] = i
            word_path = os.path.join(path, file)
            classes[file[:-4]] = get_words(word_path)
            i += 1
        for k, v in dir_dict.items():
            dict_dir[v] = k

    for key, values in classes.items():
        for value in values:
            get_vocab(value)
    print('vocab size:', len(words))
    vocab_size = len(words)
    print(words)
    for key, values in classes.items():
        label_row = [0] * len(classes)
        for value in values:
            get_one_hot(value)
            label_type.append(dir_dict[key])
            label_row[dir_dict[key]] = 1
            label.append(label_row)

vec = []
def get_vocab(short_words):
    sentence = re.sub(r'[%s]' % punctuation, '', short_words)  # 去除中文标点
    punca = set(string.punctuation)
    sentence = ''.join(pu for pu in sentence if pu not in punca)  # 去除英文标点
    vec.append([wo for wo in sentence])
    for word in sentence:#jieba.cut(sentence,cut_all=True):
        if word not in words:
            words.append(word)



#using one hot code as the vector representation
def get_one_hot(value):
    bag = []
    for word in words:
        bag.append(1) if word in value else bag.append(0)
    training_Data.append(np.asarray(bag))


get_data()
data_train=[]
for i in range(len(training_Data)):
    data_train.append([training_Data[i],label[i],label_type[i]])
random.shuffle(data_train)
train = data_train[:350]
test = data_train[350:]
t = [i[0] for i in train]
type = [i[2] for i in train]
l = [j[1] for j in train]
t_t = [i[0] for i in test]
l_l = [j[1] for j in test]
ty_ty = [i[2] for i in test]

'''
from gensim.models import Word2Vec

print(vec)
def _w2v_train(data,path):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec(data, size=100, window=3, hs=1)
    model.save(path)

_w2v_train(vec,w2v_path)
model = Word2Vec.load(w2v_path)
print(model.wv.n_similarity('已经买了','以再说'))

#print(np.shape(train_label))

'''