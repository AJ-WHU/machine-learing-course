from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# 获取训练数据，调用了sklearn自带的fetch_20newsgroups用于导入数据。
# 若下载过慢，需要按照https://www.cnblogs.com/funykatebird/articles/11733952.html所给教程进行配置，可以实现从本地导入
train_set = fetch_20newsgroups(subset='train')
test_set = fetch_20newsgroups(subset='test')

# 停用词设置
from nltk.corpus import stopwords
stopword_nltk = stopwords.words('english')  # nltk提供的停用词，有179个
with open('stopwords.txt') as f:
    stopword = f.read().split()
stop_word = set(stopword).union(set(stopword_nltk))# 取已有停用词和nltk所给停用词的并集
stop_word = list(stop_word)

from sklearn.metrics import classification_report
import numpy as np

# 使用Tfidf进行文本向量化
feature_num = 25000 # 特征数量选择
vectorizer = TfidfVectorizer(stop_words = stop_word, max_features = feature_num)
train_x = vectorizer.fit_transform(train_set.data)
test_x = vectorizer.transform(test_set.data)
Y_train = train_set.target
Y_test = test_set.target
X_train = train_x.toarray()
X_test = test_x.toarray()

# 这样之后，X_train存储训练集的特征矩阵，Y_Ttrain存储训练集的结果，剩下两个以此类推。

# 以上预处理部分，和感知机是一致的。

# 以下是训练部分
import numpy as np
p_class = np.zeros(20) # 1 * 20的向量，第i个元素存储第i类样本在总训练集中的比例
p_feature_now_class = np.zeros((feature_num,20)) # 1 * 20 的向量，即P（feature|Class）
L = 1 # 拉普拉斯平滑参数

# 开始训练
for now_class in range(20): # 遍历每个类
    now_class_rows = (Y_train == now_class) # now_class_rows存储当前类在Y_train中的下标
    X_train_now = X_train[now_class_rows] #提取X_train中属于当前类别的元素
    Y_train_now = Y_train[now_class_rows] #提取Y_train中属于当前类别的元素
    p_class[now_class] = len(Y_train_now)/len(Y_train) * 10000 #计算当前类别的P（Class）,乘上10000防止下溢
    X_train_now_T = X_train_now.T
    all_weight = np.sum(X_train_now) #计算当前类别的特征总权重
    for feature in range(feature_num): #遍历每个特征
        all_weight_of_this_feature = np.sum(X_train_now_T[feature]) #计算当前特征在当前类别中的总权重
        p_feature_now_class[feature][now_class] = (all_weight_of_this_feature + L)/(all_weight + L * feature_num) *10000
        # 计算P（feature|Class），L用于拉普拉斯平滑，乘上10000防止下溢
        
#以下是预测部分
y_pred = [] # 存储最终预测结果
for test_sample in X_test: # 遍历测试样本
    p_belong_to_now_class = np.zeros(20) # 1 * 20 的向量，第i个元素代表样本属于第i个类别的概率
    for now_class in range(20): # 遍历每个类别
        p_belong_to_now_class[now_class] = np.log(p_class[now_class]) 
        # 现在p_belong_to_now_class[now_class]里面存的是P（Class）的log值
        for feature in range(feature_num):# 遍历每个特征
            p_belong_to_now_class[now_class] += np.log(p_feature_now_class[feature][now_class] * test_sample[feature] + 1)
        # 这样之后，p_belong_to_now_class[now_class]里的第i个元素就是样本属于第i个类别的概率。+1是防止出现0以致无法取log
    I = np.argmax(p_belong_to_now_class) # 极大似然估计，取概率最大的类作为估计值
    y_pred.append(I) # 将该样本的最终预测结果加入y_pred

from sklearn.metrics import classification_report
print("在特征数为")
print(feature_num)
print("拉普拉斯平滑参数为")
print(L)
print("下的结果为")
print(classification_report(Y_test,y_pred))
