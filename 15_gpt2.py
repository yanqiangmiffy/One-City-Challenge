# 方案参考链接 https://www.kaggle.com/tanulsingh077/deep-learning-for-nlp-zero-to-transformers-bert#BERT-and-Its-Implementation-on-this-Competition
# tranformers模型 https://huggingface.co/models
# 导入包
import re
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from transformers import BertTokenizer, TFBertModel
import transformers
from tokenizers import BertWordPieceTokenizer
import pandas as pd
from tqdm import tqdm
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# 加载数据

def get_clean(text):
    r1 = '[a-zA-Z]+'
    text = re.sub(r1, '', text)
    return text[:300]
train=pd.read_csv('data/train_set.csv')
test=pd.read_csv('data/test_set.csv')
train['text']=train['text'].apply(lambda x:get_clean(x))
test['text']=test['text'].apply(lambda x:get_clean(x))
print(test['text'])
print(train.shape,test.shape)
print(train.head())


def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    """
    Encoder for encoding the text into sequence of integers for BERT Input
    """
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(length=maxlen)
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)



AUTO = tf.data.experimental.AUTOTUNE


# Configuration
EPOCHS = 5
BATCH_SIZE = 16 
MAX_LEN = 128

# First load the real tokenizer
tokenizer = BertTokenizer.from_pretrained('mymusise/gpt2-medium-chinese')
#tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
#tokenizer =transformers.DistilBertTokenizer.from_pretrained('bert-base-chinese')
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
print(fast_tokenizer)

x_train = fast_encode(train.text.astype(str), fast_tokenizer,
        maxlen=MAX_LEN)
x_valid = fast_encode(train.text.astype(str), fast_tokenizer,
        maxlen=MAX_LEN)
x_test = fast_encode(test.text.astype(str), fast_tokenizer, maxlen=MAX_LEN)

y_train = train.label.values
y_valid = train.label.values


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(BATCH_SIZE)
)

def build_model(transformer, max_len=512):
    """
    function for training the BERT model
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(20, activation='softmax')(cls_token)
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'])
    return model



transformer_layer = (TFBertModel.from_pretrained('bert-base-chinese',return_dict=True))
#transformer_layer =(TFBertModel.from_pretrained('hfl/chinese-roberta-wwm-ext',return_dict=True)) 
model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()

checkpoint=ModelCheckpoint(
                'weights.h5',
                monitor='val_sparse_categorical_accuracy',
                save_weights_only=True,
                save_best_only=True,
                verbose=1,
                mode='max'),

n_steps = x_train.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    #validation_split=0.3,
    callbacks=checkpoint,
    validation_data=valid_dataset,
    validation_steps=n_steps//2,
    epochs=EPOCHS
)

predictions=model.predict(test_dataset, verbose=1)
np.save("result/15_pred.npy",predictions)
predictions=np.argmax(predictions,axis=1)

train = pd.read_csv('data/answer_train.csv')
df_label = pd.DataFrame({'label': train.label.value_counts(normalize=True).index.tolist(),
                         'label_n': [i for i in range(train.label.nunique())]})

sub = pd.read_csv('data/submit_example_test1.csv')[['filename']]
sub['label_n'] = predictions
sub = pd.merge(sub, df_label, on='label_n', how='left')
sub.drop(['label_n'], axis=1, inplace=True)
sub.to_csv('result/15_gpt2.csv', index=False)
