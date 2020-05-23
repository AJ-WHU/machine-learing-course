import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv('train-balanced-sarcasm.csv') # 读取训练集数据
train_df.head()
train_df.info()
train_df.dropna(subset=['comment'], inplace=True)
train_df['label'].value_counts()
train_texts, valid_texts, y_train, y_valid = train_test_split(train_df['comment'], train_df['label'], random_state=17)
    
train_df.loc[train_df['label'] == 1, 'comment'].str.len().apply(np.log1p).hist(label='sarcastic', alpha=.5) 
train_df.loc[train_df['label'] == 0, 'comment'].str.len().apply(np.log1p).hist(label='normal', alpha=.5) # 分类，normal为0,sarcastic为1
plt.legend()

sub_df = train_df.groupby('subreddit')['label'].agg([np.size, np.mean, np.sum])
sub_df.sort_values(by='sum', ascending=False).head(10)

# 使用 tf-idf 提取文本特征
tf_idf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)
# 建立逻辑回归模型
logit = LogisticRegression(C=1, n_jobs=4, solver='lbfgs',random_state=17, verbose=1)
# 使用 sklearn pipeline 封装 2 个步骤
tfidf_logit_pipeline = Pipeline([('tf_idf', tf_idf),('logit', logit)])
tfidf_logit_pipeline.fit(train_texts, y_train)
valid_pred = tfidf_logit_pipeline.predict(valid_texts)
accuracy_score(y_valid, valid_pred)
