from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

#获取训练数据，调用了sklearn自带的fetch_20newsgroups用于导入数据。
#若下载过慢，需要按照https://www.cnblogs.com/funykatebird/articles/11733952.html所给教程进行配置，可以实现从本地导入
train_set = fetch_20newsgroups(subset='train')
test_set = fetch_20newsgroups(subset='test')

#停用词设置
from nltk.corpus import stopwords
stopword_nltk = stopwords.words('english')  #nltk提供的停用词，有179个
with open('stopwords.txt') as f:
    stopword = f.read().split()
stop_word = set(stopword).union(set(stopword_nltk))#取已有停用词和nltk所给停用词的并集
stop_word = list(stop_word)

from sklearn.metrics import classification_report
import numpy as np

# 使用Tfidf进行文本向量化
feature_num = 25000 #特征数量选择
vectorizer = TfidfVectorizer(stop_words = stop_word, max_features = feature_num)
train_x = vectorizer.fit_transform(train_set.data)
test_x = vectorizer.transform(test_set.data)
Y_train = train_set.target
Y_test = test_set.target
X_train = train_x.toarray()
X_test = test_x.toarray()

#这样之后，X_train存储训练集的特征矩阵，Y_Ttrain存储训练集的结果，剩下两个以此类推。

#以下为训练部分

gram = np.dot(X_train, X_train.T) #引入gram矩阵加快运算速度

a_all = []
w_all = []
b_all = [] #存储 20 个子感知机模型的 a、b 和 w

for kind in range(20): #遍历 20 个类
    samples_number = len(Y_train) #训练集样本数目
    Y_train_now = np.zeros(samples_number) 
    for i in range(samples_number):
        if Y_train[i] == kind:
            Y_train_now[i] = 1
        else:
            Y_train_now[i] = -1
    #这样之后，Y_train_now里面存储的只有 1 和 -1 ，是当前类为 1 ，不是当前类为 -1 
    
    a = np.zeros(samples_number)
    b = 0
    condition = 0
    lr = 1
    #初始化当前的 a、b和学习率lr
    train_num = 10
    #train_num 为训练轮数
    
    
    for j in range(train_num):
        for i in range(samples_number): #每轮都遍历一遍所有样本
            condition = np.dot(a*Y_train_now, gram[i]) #感知机对偶形式，使用gram矩阵加快运算速度
            condition = (condition + b) * Y_train_now[i]
            if condition <= 0:
                a[i] += lr
                b += lr * Y_train_now[i] #梯度下降
                
    w = np.dot(a*Y_train_now,X_train) #训练结束后求出此次的w
    a_all.append(a)
    w_all.append(w)
    b_all.append(b)
    
#以下为预测部分
y_pred = [] 
res = []
for x in X_test: #遍历每个测试样本
    for every_w in w_all: 
        tmp = np.dot(every_w,x.T)/np.linalg.norm(every_w)
        res.append(tmp) #计算该样本到每个超平面的距离，并保存在res中
    y_pred.append(np.argmax(res)) #取使得距离最大且为正的超平面所对应的类，作为该样本的预测结果
    res = []

print("此时特征数为")
print(feature_num)
print("训练轮数为")
print(train_num)
print("学习率为")
print(lr)
print("得到的结果是：")
print(classification_report(Y_test,y_pred))
