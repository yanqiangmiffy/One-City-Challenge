import warnings
warnings.simplefilter('ignore')
from sklearn.metrics import accuracy_score
import re
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs



def get_clean(text):
    r1 = '[a-zA-Z]+'
    text = re.sub(r1, '', text)
    text = text.replace('\\', '')
    return text[:300]
train_df=pd.read_csv('data/train_set_v3.csv')
test=pd.read_csv('data/test_set_v3.csv')
train_df['text']=train_df['text'].apply(lambda x:get_clean(x))
test['text']=test['text'].apply(lambda x:get_clean(x))
print(test['text'])
print(train_df.shape,test.shape)
print(train_df.head())


train_tmp = pd.read_csv('data/answer_train.csv')
df_label = pd.DataFrame({'label': train_tmp.label.value_counts(normalize=True).index.tolist(),
                         'label_n': [i for i in range(train_tmp.label.nunique())]})


train_df = train_df.sample(frac=1., random_state=1024)

eval_df = train_df[54000:]
train_df = train_df[:54000]



model_args = ClassificationArgs()

model_args.max_seq_length = 160
model_args.train_batch_size = 32
model_args.num_train_epochs = 5
model_args.fp16 = False
model_args.evaluate_during_training = True
model_args.overwrite_output_dir = True



model = ClassificationModel(
        "bert",
        "hfl/chinese-roberta-wwm-ext",
        #"bert",
        #"voidful/albert_chinese_base",
        #"hfl/chinese-roberta-wwm-ext-large",
        num_labels=len(df_label),
        args=model_args)



model.train_model(train_df, eval_df=eval_df)

result, _, _ = model.eval_model(eval_df, acc=accuracy_score)
print(result)


data = []
for i, row in test.iterrows():
    data.append(row['text'])


predictions, raw_outputs = model.predict(data)



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



sub = pd.read_csv('data/submit_example_test1.csv')[['filename']]
sub['label'] = predictions
sub['label']=sub['label'].map(label_index_inverse)

print(sub.shape)
print(sub.head(10))

np.save('result/10_chinese-roberta-wwm-ext_p3.npy', raw_outputs)
sub.to_csv('result/10_chinese-roberta-wwm-ext_p3.csv', index=False)
