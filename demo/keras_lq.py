import numpy as np

np.random.seed(1337)

from keras.models import Sequential
from keras.layers import Dense,Dropout
import matplotlib.pyplot as plt
from robot_emotion import *
from keras.layers import Embedding
from keras.layers import LSTM
from keras.utils.np_utils import to_categorical

X = np.array(t)
np.random.shuffle(X)
Y = np.array(l)
plt.show()

x_train, y_train = X, Y
x_test, y_test = np.array(t_t),np.array(l_l)

model = Sequential()
model.add(Dense(128, input_dim=307, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print('training............................')

cost = model.fit(x_train, y_train,epochs=1000)

print('testing...........')
cost = model.evaluate(x_test, y_test)

print('test cost', cost)
#w, b = model.layers[0].get_weights()

#print('weight:', w, 'bias:', b)

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
def pre(word):
    wo = []
    for w in word:
        wo.append(w)
    bag = [0] * len(words)
    for s in wo:
        for index, word in enumerate(words):
            if word == s:
                bag[index] = 1
    return (np.array(bag))
y_pr=[]
def get_tetstlabel(X,Y):
    for t in X:
        test_row = [0] * len(dict_dir)
        y_pre=model.predict(np.array(t).reshape(1,307))
        result = [[i, v] for i, v in enumerate(y_pre[0])]
        result.sort(key=lambda x: x[1], reverse=True)
        test_row[result[0][0]] = 1
        #print(result[0][0],test_row)
        y_pr.append(test_row)
        #print(y_pre,result)

        # print(result[0][1])
        # return_results = dict_dir.get(result[0][0])
        # print(return_results)


get_tetstlabel(t_t,l_l)
f=get_f1_score(l_l,y_pr)
print(f)
