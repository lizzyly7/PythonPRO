import numpy as np
from nelq import *
from sklearn.metrics import f1_score
from robot_emotion import t_t,l_l


import json

#ERROR_THREADHOLE = 0.2


def classify(sentence, show_details=False,cut_sen=False):
    test_row = [0] * len(dict_dir)
    result = think(sentence, show_details,cut_sen=cut_sen)

    result = [[i, r] for i, r in enumerate(result) ]#if r > ERROR_THREADHOLE]
    result.sort(key=lambda x: x[1], reverse=True)
    #test_row[result[0][0]] = 1
    return_results = dict_dir.get(result[0][0])
    #print(return_results)

    #print(r'%s \n classification:%s' % (sentence, return_results))
    return return_results

y_pre=[]
def pre_test(t_t,l_l):
    for index,sentence in enumerate(t_t):
        y_pre.append(classify(sentence,cut_sen=False))
    assert len(y_pre) == len(l_l),'the numbe rshould be same'
    #f1 = f1_score(Y,y_pre)


def get_test(test_data,test_label):
    test_row = [0]*len(dict_dir)
    test =[]
    for word in test_data:
        test_row[classify(word)] = 1
        test.append(test_row)
    assert len(test) == len(test_label),'wrong for label length'
    return sum(int(test_label[i] == test[i]) for i in range(len(test_label)))
#because of the unbalance data which is a bug for this model, bue using f1_score as the stardard
# to judge the performance is quite acceptable
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

def test():
    with open('D:\data\data_custumer.txt','r',encoding ='utf-8') as file:
        data = file.readlines()
        s =[]
    for dat in data[:100]:
        result = classify(dat,show_details=False,cut_sen=True)
        s.append(result)
    da_path = r'C:\Users\Administrator\Desktop\test_one.csv'
    out = open(da_path, 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    for i in zip(set(data[:100]), s):
        csv_write.writerow(i)
test()

#pre_test(t_t,l_l)
#y_true = ([[1, 1, 0, 0, 1], [1, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
#y_hat = ([[0, 1, 1, 1, 1], [1, 0, 0, 1, 1], [1, 0, 1, 0, 0]])
#f =get_f1_score(l_l,y_pre)


    
