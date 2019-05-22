

from true_test import *
from operator import itemgetter
from keras.models import load_model
import pandas as pd
#

train_s1 =[]
train_s2 =[]
cos_s = []
file_path = r'C:\Users\Administrator\Desktop\qa.csv'
try:
    data_all = pd.read_csv(file_path,header=None,names=["id","question_1","id_extend","id_2","answer"],encoding='utf-8')
except TimeoutError as t:
    print('load file wrong')
for ques in data_all['question_1']:
    get_vocab(ques)
    #train_s1.append(get_one_code(ques))
for qu in data_all['answer']:
    get_vocab(qu)
print('vocab size:', len(words))
vocab_path = r'C:\Users\Administrator\Desktop\vocab.json'
vocab_w(vocab_path)
    #train_s2.append(get_one_code(qu))
for que in data_all['question_1']:
    train_s1.append(get_one_code(que))
for q in data_all['question_2']:
    train_s2.append(get_one_code(q))
assert len(train_s2) == len(train_s1), 'two type of data should be same length'

for i in range(len(train_s1)):
    cos_s.append(1) if cosine_sim(train_s1[i],train_s2[i]) > 0.2 else cos_s.append(0)

assert len(cos_s)==len(train_s2),'the len(cos) should same to len(train_s1)'

