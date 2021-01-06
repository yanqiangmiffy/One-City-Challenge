import numpy as np
import pandas as pd
import os

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

pred = None
for file in os.listdir('result/'):
    if file.endswith('.npy'):
        if pred is None:
            pred = np.load('result/{}'.format(file))
        else:
            pred += np.load('result/{}'.format(file))

predictions = np.argmax(pred, axis=1)
sub = pd.read_csv('data/submit_example_test1.csv')[['filename']]
sub['label'] = predictions
sub['label'] = sub['label'].map(label_index_inverse)
print(sub.head(10))
sub.to_csv('result/ensemble_mean.csv', index=False)
