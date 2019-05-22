import os
import json
from robot_emotion import dict_dir,dir_dict,classes,words,data_train,t,l
from prediction import classify
from sim_code import pre,cosine_sim
import csv
import numpy as np
import logging


sys_qa_path = r'C:\Users\Administrator\Desktop\sys_qa(2).json'
sentence = {}
def file_reader(path):
    son =[]
    if os.path.exists(path):
        with open(path,'r',encoding='utf-8') as files:
            lines = files.read()
            file = json.loads(lines)
            for fil in file['RECORDS']:
                son.append(fil['question'])
    sentence['sen1']= son
    return sentence

def pre_label_bp():
    y = []
    for value in sentence['sen1']:
        y.append(classify(value,cut_sen=True))
    return y

label = []
def cosine_sim_test():
    #first: one_hot code second: cosine_sim
    for value in sentence['sen1']:
        y = []
        for data in t:
            if 1 not in pre(value):
                y.append(0)
            else:
                y.append(cosine_sim(pre(value), data))

        score = y.index(max(y))

        label.append(dict_dir[l[score].index(max(l[score]))])
    print(label)
    return label

s = file_reader(sys_qa_path)

path = r'C:\Users\Administrator\Desktop\sys_qa_cosine1.csv'
files = open(path,'w',newline='')
csv_write = csv.writer(files,dialect='excel')
csv_write.writerows([i for i in zip(s['sen1'],cosine_sim_test())])
#csv_write.writerows([i for i in zip(s['sen1'],pre_label_bp())])
