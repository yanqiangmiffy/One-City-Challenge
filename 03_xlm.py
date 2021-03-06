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

model_args.max_seq_length = 150
model_args.train_batch_size = 32
model_args.num_train_epochs = 5
model_args.fp16 = False
model_args.evaluate_during_training = True
model_args.overwrite_output_dir = True



model = ClassificationModel(
        "xlnet",
        "hfl/chinese-xlnet-base",
        num_labels=len(df_label),
        args=model_args)



model.train_model(train_df, eval_df=eval_df)

result, _, _ = model.eval_model(eval_df, acc=accuracy_score)
print(result)


data = []
for i, row in test.iterrows():
    data.append(row['text'])


predictions, raw_outputs = model.predict(data)


sub = pd.read_csv('data/submit_example_test1.csv')[['filename']]
sub['label_n'] = predictions
sub = pd.merge(sub, df_label, on='label_n', how='left')
sub.drop(['label_n'], axis=1, inplace=True)

print(sub.shape)
print(sub.head(10))

np.save('result/03_chinese-xlnet-base_p3.npy', raw_outputs)
sub.to_csv('result/03_chinese-xlnet-base_p3.csv', index=False)
