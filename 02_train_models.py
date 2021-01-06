import re
import numpy as np
import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import accuracy_score
import gc

def get_clean(text):
    r1 = '[a-zA-Z]+'
    text = re.sub(r1, '', text)
    text = text.replace('\\', '')
    return text[:300]


train_df = pd.read_csv('data/train_set.csv')
test = pd.read_csv('data/test_set.csv')
train_df['text'] = train_df['text'].astype(str)
test['text'] = test['text'].astype(str)
print(test.head())
label_index = {
    '工业': 0,
    '文化休闲': 1,
    '教育科技': 2,
    '医疗卫生': 3,
    '文秘行政': 4,
    '生态环境': 5,
    '城乡建设': 6,
    '农业畜牧业': 7,
    '经济管理': 8,
    '交通运输': 9,
    '政法监察': 10,
    '财税金融': 11,
    '劳动人事': 12,
    '旅游服务': 13,
    '资源能源': 14,
    '商业贸易': 15,
    '气象水文测绘地震地理': 16,
    '民政社区': 17,
    '信息产业': 18,
    '外交外事': 19}

train_df['label'] = train_df['label'].map(label_index)

# train_df['text'] = train_df['text'].apply(lambda x: get_clean(x))
# test['text'] = test['text'].apply(lambda x: get_clean(x))
print(test['text'])
print(train_df.shape, test.shape)
print(train_df.head())

train_df = train_df.sample(frac=1., random_state=1024)

eval_df = train_df[54000:]
train_df = train_df[:54000]

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

models = [
    ('bert', 'hfl/chinese-roberta-wwm-ext'),
    ('xlnet', 'hfl/chinese-xlnet-base'),
    ('bert', 'schen/longformer-chinese-base-4096'),
    ('bert', 'voidful/albert_chinese_base'),
    ('bert', 'clue/roberta_chinese_base'),
    ('electra', 'hfl/chinese-electra-base-discriminator'),
    # ('bert', 'hfl/chinese-roberta-wwm-ext')
]

for i in range(len(models)):
    print("training {}".format(models[i][1]))
    model_args = ClassificationArgs()
    model_args.max_seq_length = 128
    model_args.train_batch_size = 64
    model_args.num_train_epochs =3
    model_args.fp16 = False
    model_args.evaluate_during_training = False
    model_args.overwrite_output_dir = True

    model_type = models[i][0]
    model_name = models[i][1]
    print('./outputs' + '/' + model_name.split('/')[1])
    model_args.cache_dir = './caches' + '/' + model_name.split('/')[1]
    model_args.output_dir = './outputs' + '/' + model_name.split('/')[1]

    print("==========================", models[i], '======================')
    model = ClassificationModel(
        model_type,
        model_name,
        num_labels=len(label_index_inverse),
        args=model_args)

    model.train_model(train_df, eval_df=eval_df)
    result, _, _ = model.eval_model(eval_df, acc=accuracy_score)
    print(result)
    data = []
    for index, row in test.iterrows():
        data.append(str(row['text']))

    predictions, raw_outputs = model.predict(data)

    sub = pd.read_csv('data/submit_example_test2.csv')[['filename']]
    sub['label'] = predictions
    sub['label'] = sub['label'].map(label_index_inverse)

    print(sub.shape)
    print(sub.head(10))
    result_name = models[i][1].split('/')[1]
    print('{}_{}.npy'.format(i + 1, result_name))
    np.save('result/{}_{}.npy'.format(i + 1, result_name), raw_outputs)
    sub.to_csv('result/{}_{}.csv'.format(i + 1, result_name), index=False)
    del model
    gc.collect()
