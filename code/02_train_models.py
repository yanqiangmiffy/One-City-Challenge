import re
import numpy as np
import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import accuracy_score


def get_clean(text):
    r1 = '[a-zA-Z]+'
    text = re.sub(r1, '', text)
    text = text.replace('\\', '')
    return text[:300]


train_df = pd.read_csv('data/train_set_v3.csv')
test = pd.read_csv('data/test_set_v3.csv')
train_df['text'] = train_df['text'].apply(lambda x: get_clean(x))
test['text'] = test['text'].apply(lambda x: get_clean(x))
print(test['text'])
print(train_df.shape, test.shape)
print(train_df.head())

train_tmp = pd.read_csv('data/answer_train.csv')

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
]

for i in range(len(models)):
    print("training {}".format(models[i][1]))
    model_args = ClassificationArgs()
    model_args.max_seq_length = 150
    model_args.train_batch_size = 32
    model_args.num_train_epochs = 5
    model_args.fp16 = False
    model_args.evaluate_during_training = True
    model_args.overwrite_output_dir = True
    model_args.cache_dir = './caches'
    model_args.output_dir = './outputs'

    model_type = models[i][0]
    model_name = models[i][1]

    model = ClassificationModel(
        model_type,
        model_name,
        num_labels=len(label_index_inverse),
        args=model_args)

    model.train_model(train_df, eval_df=eval_df)
    result, _, _ = model.eval_model(eval_df, acc=accuracy_score)

    data = []
    for i, row in test.iterrows():
        data.append(row['text'])

    predictions, raw_outputs = model.predict(data)

    sub = pd.read_csv('data/submit_example_test1.csv')[['filename']]
    sub['label'] = predictions
    sub['label'] = sub['label'].map(label_index_inverse)

    print(sub.shape)
    print(sub.head(10))
    result_name = models[i][1].split('/')[1]
    np.save('result/{}_{}.npy'.format(i, result_name), raw_outputs)
    sub.to_csv('result/{}}_{}.csv'.format(i, result_name), index=False)
