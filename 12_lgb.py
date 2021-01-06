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

tqdm.pandas()
train_df = pd.read_csv('data/train_set.csv')
test_df = pd.read_csv('data/test_set.csv')


def get_clean(text):
    r1 = '[a-zA-Z]+'
    text = re.sub(r1, '', text)
    # text = re.sub(r'(年)\1+', r'\1', text)
    text = text.replace('\\', '')
    return text[:1500]


def tokenizer(text):
    text=str(text)
    text = " ".join([w for w in jieba.cut(text) if w][:300])
    text = " ".join(text.split())
    return text


#train_df['text'] = train_df['text'].apply(lambda x: get_clean(x))
#test_df['text'] = test_df['text'].apply(lambda x: get_clean(x))
train_df['text'] = train_df['text'].progress_apply(tokenizer)
test_df['text'] = test_df['text'].progress_apply(tokenizer)

df_text = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
tfidf = TfidfVectorizer(max_df=0.98, min_df=2, ngram_range=(1, 2),
        max_features=160000, sublinear_tf=True)
tfidf.fit(df_text.text.values)
X = tfidf.transform(train_df.text.values)  # (60000, 61330)
print(X.shape)
Y = train_df.label.values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)

params = {
    'learning_rate': 0.05,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'objective': 'multiclass',
    'n_estimators': 600,
    'metric': 'multi_logloss',
    'random_state': 2020,
    'max_depth': 20,
    'verbose': 0,
    'min_child_samples': 63,
    'lambda_l1': 0.01,
    'lambda_l2': 0,
    'num_class': 20
}

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'n_estimators': 100000,
    'learning_rate': 0.1,
    'random_state': 2948,
    'bagging_freq': 8,
    'bagging_fraction': 0.80718,
    # 'bagging_seed': 11,
    'feature_fraction': 0.38691,  # 0.3
    'feature_fraction_seed': 11,
    'max_depth': 9,
    'min_data_in_leaf': 40,
    'min_child_weight': 0.18654,
    "min_split_gain": 0.35079,
    'min_sum_hessian_in_leaf': 1.11347,
    'num_leaves': 29,
    'num_threads': 4,
    "lambda_l1": 0.55831,
    'lambda_l2': 1.67906,
    'cat_smooth': 10.4,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    # 'verbosity': 1,
}
models = []

for index_, (train_ind, test_ind) in enumerate(skf.split(X, Y)):
    print(f"Round {index_} begin.")

    X_train, X_test = X[train_ind], X[test_ind]
    Y_train, Y_test = Y[train_ind], Y[test_ind]

    lgb = lightgbm.LGBMClassifier(**lgb_params)
    lgb.fit(X_train, Y_train, eval_set=(X_test, Y_test), early_stopping_rounds=200)

    f1_train = f1_score(Y_train, lgb.predict(X_train), average='macro')
    f1_test = f1_score(Y_test, lgb.predict(X_test), average='macro')

    print(f"train score is {f1_train}; test score is {f1_test}; ")

    models.append([
        index_,
        lgb,
        ('train_f1_sc', f1_train),
        ('test_f1_sc', f1_test)
    ])

    print("=" * 64)

test = tfidf.transform(test_df.text.values)

pred = np.zeros((len(test_df), 20))
t_X = []
for m in tqdm(models):
    pred += m[1].predict_proba(test) / 5

np.save("lgb.npy", pred)

label_index_inverse = {
    0: '工业',
    1: '文化休闲',
    2: '教育科技',
    3: '医疗卫生',
    4: '文秘行政',
    5: '生态环境',
    6: '城乡建设',
    7: '农业畜牧业',
    8: '经济管理',
    9: '交通运输',
    10: '政法监察',
    11: '财税金融',
    12: '劳动人事',
    13: '旅游服务',
    14: '资源能源',
    15: '商业贸易',
    16: '气象水文测绘地震地理',
    17: '民政社区',
    18: '信息产业',
    19: '外交外事'}
sub = pd.read_csv('data/submit_example_test2.csv')[['filename']]
sub['label'] = np.argmax(pred, axis=1)
sub['label'] = sub['label'].map(label_index_inverse)
print(sub.shape)
sub.to_csv('result/lgb.csv', index=False)

