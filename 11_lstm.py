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

warnings.filterwarnings('ignore')

tqdm.pandas()
seed = 2020
np.random.seed(seed)
tf.random.set_seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def build_model(emb, seq_len):
    '''
    构建模型
    '''
    emb_layer = tf.keras.layers.Embedding(
        input_dim=emb.shape[0],
        output_dim=emb.shape[1],
        input_length=seq_len
    )
    print(emb.shape)

    seq = tf.keras.layers.Input(shape=(seq_len,))
    seq_emb = emb_layer(seq)

    seq_emb = tf.keras.layers.SpatialDropout1D(rate=0.2)(seq_emb)

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True))(seq_emb)
    lstm_avg_pool = tf.keras.layers.GlobalAveragePooling1D()(lstm)
    lstm_max_pool = tf.keras.layers.GlobalMaxPooling1D()(lstm)
    x = tf.keras.layers.Concatenate()([lstm_avg_pool, lstm_max_pool])

    x = tf.keras.layers.Dropout(0.2)(tf.keras.layers.Activation(activation='relu')(
        tf.keras.layers.BatchNormalization()(tf.keras.layers.Dense(1024)(x))))
    out = tf.keras.layers.Dense(20, activation='softmax')(x)

    model = tf.keras.Model(inputs=seq, outputs=out)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()
    return model


class Evaluator(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.val_f1 = 0.
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def evaluate(self):
        y_true = self.y_val
        y_pred = self.model.predict(self.x_val).argmax(axis=1)
        f1 = f1_score(y_true, y_pred, average='macro')
        return f1

    def on_epoch_end(self, epoch, logs=None):
        val_f1 = self.evaluate()
        if val_f1 > self.val_f1:
            self.val_f1 = val_f1
        logs['val_f1'] = val_f1
        print(f'val_f1: {val_f1:.5f}, best_val_f1: {self.val_f1:.5f}')


if __name__ == "__main__":
    '''
    数据加载
    '''
    train_df = pd.read_csv('data/train_set.csv')
    test_df = pd.read_csv('data/test_set.csv')


    def get_clean(text):
        r1 = '[a-zA-Z]+'
        text = re.sub(r1, '', text)
        # text = re.sub(r'(年)\1+', r'\1', text)
        text = text.replace('\\', '')
        return text


    def tokenizer(text):
        text=" ".join([w for w in jieba.cut(text) if w])
        text=" ".join(text.split())
        return text


    train_df['text'] = train_df['text'].apply(lambda x: get_clean(x))
    test_df['text'] = test_df['text'].apply(lambda x: get_clean(x))
    train_df['text'] = train_df['text'].progress_apply(tokenizer)
    test_df['text'] = test_df['text'].progress_apply(tokenizer)
    test_df['label'] = -1
    df_data = train_df.append(test_df, ignore_index=True)
    
    print(df_data.head())

    '''保留最大词数: None为不限制'''
    max_words_num = 30000
    '''序列长度'''
    seq_len = 1000
    '''embedding向量维度'''
    embedding_dim = 128
    '''序列字段名'''
    col = 'text'

    seq_path = f"./data/seqs_{max_words_num}_{seq_len}.npy"
    word_index_path = f"./data/word_index_{max_words_num}_{seq_len}.npy"

    print("begin preparing.")
    print("=" * 64)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words_num, lower=True, filters='')
    tokenizer.fit_on_texts(df_data[col].values.tolist())

    seqs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences(df_data[col].values.tolist()), maxlen=seq_len, padding='post', truncating='pre')
    word_index = tokenizer.word_index
    print(len(word_index))
    '''保存填充序列和序列字典'''
    np.save(seq_path, seqs)
    np.save(word_index_path, word_index)

    embedding = np.zeros((len(word_index) + 1, embedding_dim))
    print(embedding.shape)
    print("finish preparing.")
    print("=" * 64)

    all_index = df_data[df_data['label'] != -1].index.tolist()
    test_index = df_data[df_data['label'] == -1].index.tolist()

    '''批次大小'''
    batch_size = 128
    monitor = 'val_f1'

    '''5折训练'''
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold_id, (train_index, val_index) in enumerate(kfold.split(all_index, df_data.iloc[all_index]['label'])):
        train_x = seqs[train_index]
        val_x = seqs[val_index]

        label = df_data['label'].values
        train_y = label[train_index]
        val_y = label[val_index]

        model_path = f"./models/lstm_{fold_id}.h5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor=monitor,
            verbose=1,
            save_best_only=False,
            mode='max',
            save_weights_only=True)

        earlystopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=10, verbose=1, mode='max')
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=2, mode='max', verbose=1)
        print(embedding.shape)
        model = build_model(embedding, seq_len)
        model.fit(train_x, train_y, batch_size=batch_size, epochs=20,
                  validation_data=(val_x, val_y),
                  callbacks=[Evaluator(validation_data=(val_x, val_y)),
                             checkpoint,
                             reduce_lr,
                             earlystopping], verbose=1, shuffle=True)

    '''预测结果'''
    oof_pred = np.zeros((len(all_index), 14))
    test_pred = np.zeros((len(test_index), 14))

    '''5折结果汇总取均值'''
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold_id, (train_index, val_index) in enumerate(kfold.split(all_index, df_data.iloc[all_index]['label'])):
        model = build_model(embedding, seq_len)
        model_path = f"./models/lstm_{fold_id}.h5"
        model.load_weights(model_path)

        val_x = seqs[val_index]
        prob = model.predict(val_x, batch_size=batch_size, verbose=1)
        oof_pred[val_index] = prob

        test_x = seqs[test_index]
        prob = model.predict(test_x, batch_size=batch_size, verbose=1)
        test_pred += prob / 5

    df_oof = df_data.loc[all_index][['label']]
    df_oof['predict'] = np.argmax(oof_pred, axis=1)
    f1score = f1_score(df_oof['label'], df_oof['predict'], average='macro')
    print(f1score)

    '''保存结果'''
    np.save(f"./result/sub_5fold_lstm_{f1score}.npy", test_pred)
    np.save(f"./result/sub_5fold_lstm_{f1score}.npy", oof_pred)

    sub = pd.DataFrame()
    sub['label'] = np.argmax(test_pred, axis=1)
    sub.to_csv(f"./result/lstm_res.csv", index=False)
    print("End")

