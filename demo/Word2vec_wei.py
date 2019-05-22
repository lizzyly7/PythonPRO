import gensim
import pandas as pd
import jieba
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

path =r'C:\Users\Administrator\Desktop\PythonPRO\word2vec_from_weixin\word2vec\word2vec_wx'
zhuhu_path = r'D:\project\Q_sim\mayi_nlp\sgns.zhihu.bigram'
model = gensim.models.Word2Vec.load(path)
model1 =gensim.models.KeyedVectors.load_word2vec_format(zhuhu_path,binary=False)
#d = pd.Series(model.wv.most_similar(u'蚂蚁花呗'))
sen = '好的'
sen2 = '你是不是有病'
def lo(sentence):
    sen = [i for i in jieba.cut(sentence)]

    print(sen)
    wv = np.zeros(256)
    for se in sen:
        if se != ' ':
            try:
                wv += model.wv.get_vector(se)
            except KeyError:
                continue
        else:
            continue
    wv = wv / len(sen)
    return wv

sim = cosine_similarity(np.asarray(lo(sen)).reshape(1,256),np.asarray(lo(sen2)).reshape(1,256))
print(sim)

#c = pd.Series(model1.wv.word_vec(u'什么'))
#print(c)

#print(d)
