import os
import numpy as np
import pandas as pd
import warnings
from gensim.models import Word2Vec
from tqdm import tqdm
import random
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np
import os
from tqdm import tqdm, trange
import lightgbm
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import jieba
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import time
from pandarallel import pandarallel
pandarallel.initialize()

warnings.filterwarnings('ignore')

train_df = pd.read_csv('data/train_set.csv')
test_df = pd.read_csv('data/test_set.csv')


def get_clean(text):
    r1 = '[a-zA-Z：，〔 〕]+'
    text = re.sub(r1, '', text)
    # text = re.sub(r'(年)\1+', r'\1', text)
    text = text.replace('\\', '').replace('（','').replace('）','')
    return text


def tokenizer(text):
    text=" ".join([w for w in jieba.cut(text) if w])
    text=" ".join([w  for w in text.split()])
    return text


train_df['text'] = train_df['text'].apply(lambda x: get_clean(x))
test_df['text'] = test_df['text'].apply(lambda x: get_clean(x))
train_df['text'] = train_df['text'].parallel_apply(tokenizer)
test_df['text'] = test_df['text'].parallel_apply(tokenizer)
test_df['label'] = -1
df_data = train_df.append(test_df, ignore_index=True)
df_data['text_len']=df_data['text'].apply(lambda x :len(x.split()))
print(df_data['text_len'].describe())
print(df_data['text_len'].value_counts())

train_df.to_csv('data/train_seg.csv',index=None)
test_df.to_csv('data/test_seg.csv',index=None)
