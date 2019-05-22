from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM
#from robot_emotion import *
import matplotlib.pyplot as plt
from true_test import *

X = np.array()
np.random.shuffle(X)
Y = np.array(l)
plt.show()

x_train, y_train = X, Y
x_test, y_test = np.array(t_t),np.array(l_l)

model = Sequential()
model.add((Embedding(input_dim=307, output_dim = 8)))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(8, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Train the model, iterating on the data in batches of 32 samples
model.fit(x_train, y_train, epochs=30, batch_size=50)
score = model.evaluate(x_test,y_test)
print(score)


'''
x = pre('我看一下，我没打算买')
print(x)
y_pre = model.predict(x.reshape(1, 307))
result = [[i,v] for i,v in enumerate(y_pre[0])]
result.sort(key= lambda x:x[1],reverse=True)
print(result[0][1])
return_results = dict_dir.get(result[0][0])
print(return_results)
'''
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

get_tetstlabel(t_t,l_l)
f = get_f1_score(l_l,y_pr)
print(f)