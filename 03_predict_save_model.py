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


test = pd.read_csv('data/test_set_v3.csv')
#test['text'] = test['text'].apply(lambda x: get_clean(x))
print(test['text'])
test['text']=test['text'].astype(str)

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

model = ClassificationModel(
    "bert", "chinese-roberta-wwm-ext/outputs/best_model"
)
data = []
for index, row in test.iterrows():
    data.append(row['text'])

predictions, raw_outputs = model.predict(data)
sub = pd.read_csv('data/submit_example_test2.csv')[['filename']]
sub['label'] = predictions
sub['label'] = sub['label'].map(label_index_inverse)

print(sub.shape)
print(sub.head(10))
result_name = models[0][1].split('/')[1]
np.save('result/{}_{}.npy'.format(0, result_name), raw_outputs)
sub.to_csv('result/{}_{}.csv'.format(0, result_name), index=False)

