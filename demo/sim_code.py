from robot_emotion import *
import jieba

import math
from scipy.spatial.distance import pdist
import numpy as np
#from data_custumer import data
import csv

def cosine_sim(x, y):
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return cos


def pre(sentence):
    bag = []
    sen = [i for i in jieba.cut(sentence,cut_all=True)]
    for word in words:
        bag.append(1) if word in sen else bag.append(0)
    return np.array(bag)

y_pre=[]
def get_sim(x):
    y_l = []
    if 1 not in x:
        print('unseen words',x)

    else:
        for y in range(len(train)):
            y_l.append(cosine_sim(x, train[y][0]))
        #print(y_l)

        score = y_l.index(max(y_l))

        y_pre.append(data_train[score][1])
        #print(dict_dir.get((data_train[score][1]).index(1)))

def get_f1_score(Y,y_pre):
    epsilon = 1e-7
    y_pre = np.int8(y_pre)
    Y = np.int8(Y)
    tp = np.sum(Y*y_pre,axis=0)
    fp = np.sum((1-y_pre)*Y,axis=0)
    fn = np.sum(y_pre*(1-Y),axis=0)
    p = tp/(tp+fp+epsilon)
    r = tp/(tp+fn+epsilon)
    f1 = 2*p*r/(p+r+epsilon)
    f1 = np.where(np.isnan(f1), np.zeros_like(f1), f1)
    return np.mean(f1)

y_pr =[]
def get_sim_cus(word):
    y_p =[]
    if 1 not in pre(word):
        print('unseen words')
        y_pr.append('unseen')
    else:
        for y in range(len(train)):
            y_p.append(cosine_sim(pre(word), train[y][0]))
        score = y_p.index(max(y_p))

        y_pr.append(dict_dir.get(data_train[score][1].index(1)))

if __name__ == '__main__':
    '''
    for t in t_t:
        get_sim(t)
    f = get_f1_score(l_l, y_pre)
    print(f)

   
    print(np.shape(t_t))
    #x = pre('我看一下，我没打算买')
    for t in t_t:
        get_sim(t)
    f = get_f1_score(l_l,y_pre)
    print(f)
    
    '''
    with open('D:\data\data_custumer.txt','r',encoding ='utf-8') as file:
        data = file.readlines()
    for da in data:
        get_sim_cus(da)
    assert  len(y_pr)==len(data), 'num should be same'
    data_pre = {
        'sentence':data,
        'label':y_pr
    }
    import time
    start = time.time()
    #js =json.dumps(data_pre)
    d_path = 'D:\data\data_csv.csv'
    out = open(d_path, 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    for i in zip(data, y_pr):
        csv_write.writerow(i)
        e = time.time() -start
        print(e)




