import numpy as np
import time
#from first import training,output,classes,words
import nltk
import datetime
import json
import os,sys
from nltk.stem.lancaster import LancasterStemmer
import json
import datetime
from robot_emotion import *

stemmer = LancasterStemmer()

#get sigmoid function set
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

#derivate te sigmoid function:
def sigmoid_derivation(x):
    return x*(1-x)

def clean_up_sentence(sentence):
    #tokenize the pattern
    sentence_word = nltk.word_tokenize(sentence)
    #stem each words
    sentence_word = [stemmer.stem(word.lower())for word in sentence_word]
    return sentence_word


def cut_words(sentence):
    '''
    for sen in sentence:
        sen_cut.append(sen)
    return sen_cut
    '''
    sen_cut = jieba.cut(sentence)
    return sen_cut

#return bag of words array:0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details = False):
    #tokenize the pattern
    #sentence = cut_words(sentence)
    #bag of words
    #print(sentence_words)
    bag = []
    for w in words:
        bag.append(1) if w in sentence else bag.append(0)
    return (np.array(bag))

def think(sentence,show_details = False,cut_sen = False):
    if cut_sen:
        x = bow(sentence,words,show_details)
    else:
        x = sentence
    if show_details:
        print('sentence:',sentence,r'\n bow:',x)
    #input layer is our bag of words
    synapse_file = r'C:\Users\Administrator\Desktop\PythonPRO\demo\synapses_lq_da.json'
    with open(synapse_file) as data_file:
        synapse = json.load(data_file)
        synapse_0 = np.asarray(synapse['synapse0'])
        synapse_1 = np.asarray(synapse['synapse1'])
    l0 = x
    l1 = sigmoid(np.dot(l0,synapse_0))
    l2 = sigmoid(np.dot(l1,synapse_1))
    return l2


def train(X, y, neural, alpha, epochs, dropout = False, drop_precent=0.5):
    print('training with %s nerual, alpha:%s,dropout:%s%s' %(neural,str(alpha),dropout,drop_precent if dropout else''))
    print('input matrix:%sx%s outputmatrix %sx%s' %(len(X),len(X[0]),1,len(training_Data)))
    np.random.seed(1)

    last_mean_error = 1
    #initialize the weight
    synapse_0 = 2*np.random.random((len(X[0]),neural))-1
    print(np.shape(synapse_0))
    synapse_1 = 2*np.random.random((neural,len(classes)))-1
    print(np.shape(synapse_1))

    pre_synapse_0_weight_update = np.zeros_like(synapse_0)
    pre_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_prediction_count = np.zeros_like(synapse_0)
    synapse_1_prediction_count = np.zeros_like(synapse_1)

    for j in iter(range(epochs+1)):
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0,synapse_0))

        if dropout:
            layer_1 *= np.random.binomial([np.ones((len(X),nerual))],1-drop_percent)[0]*(1.0/(1-drop_precent))

        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        layer_2_error = y - layer_2
        if (j%1000) == 0 :
            if np.mean(np.abs(layer_2_error))<last_mean_error:
                print('delta after'+str(j)+'iterations'+str(np.mean(np.abs(layer_2_error))))
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print('break',np.mean(np.abs(layer_2_error)),'>',last_mean_error)
                break

        layer_2_delta = layer_2_error *sigmoid_derivation(layer_2)

        layer_1_error = layer_2_delta.dot(synapse_1.T)

        layer_1_delta = layer_1_error* sigmoid_derivation(layer_1)

        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

        if(j>0):
            synapse_0_prediction_count += np.abs(((synapse_0_weight_update>0)+0)-((pre_synapse_0_weight_update>0)+0))
            synapse_1_prediction_count += np.abs(((synapse_1_weight_update>0)+0)-((pre_synapse_1_weight_update>0)+0))

        synapse_1 +=alpha *synapse_1_weight_update
        synapse_0 +=alpha*synapse_0_weight_update
    now = datetime.datetime.now()

    #persist synapses
    synapse = {
        'synapse0':synapse_0.tolist(),'synapse1':synapse_1.tolist(),
        'datetime':now.strftime('%Y-%m-%d-%H:%M'),
        'words':words
    }
    synapse_file = 'synapses_lq_da.json'

    with open(synapse_file,'w') as outfile:
        json.dump(synapse,outfile,indent=4,sort_keys =True)
    print('saved synapse to :',synapse_file)

if __name__=='__main__':
    X=np.array(t)
    y=np.array(l)

    start_time = time.time()

    train(X,y,neural=300,alpha =0.1,epochs=10000,dropout=False,drop_precent=0.2)
    elapsed_time = time.time()-start_time
    print('processing time:',elapsed_time,'seconds')


