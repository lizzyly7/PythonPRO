import numpy as np
from functools import reduce

from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB


class Bayes:
    def __init__(self,classi_name,train_X,train_Y,test_X,test_y):
        classi_name = classi_name
        X = train_X
        y = train_Y
        x_x = test_X
        label = test_y

    def train(self):
        if self.classi_name == GaussianNB:
            classi = MultinomialNB()
        elif self.classi_name == MultinomialNB:
            classi = MultinomialNB()
        elif self.classi_name == BernoulliNB:
            classi = BernoulliNB()
        else:
            print('wrong bayes method name')
        classi.fit(self.X,self.y)

        dataset_predict_y=classi.predict(self.x_x)
        correct_predicts=(dataset_predict_y==self.label).sum()
        accuracy=100*correct_predicts/len(self.x_x)
        print('GaussianNB, correct prediction num: {}, accuracy: {:.2f}%'.format(correct_predicts,accuracy))




