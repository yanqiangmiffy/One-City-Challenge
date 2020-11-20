import warnings
from sklearn.metrics import accuracy_score
import re
import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from augmentation import aug_df
from sklearn.model_selection import train_test_split
warnings.simplefilter('ignore')


def get_clean(text):
    r1 = '[a-zA-Z]+'
    text = re.sub(r1, '', text)
    #text = re.sub(r'(年)\1+', r'\1', text)
    text = text.replace('\\', '')
    return text[:300]


train_df = pd.read_csv('data/train_set.csv')
print(train_df.shape)
#train_df = aug_df(train_df,text_len=300)
print(train_df.shape)

test = pd.read_csv('data/test_set.csv')
train_df['text'] = train_df['text'].apply(lambda x: get_clean(x))
test['text'] = test['text'].apply(lambda x: get_clean(x))
print(test['text'])
print(train_df.head())

train_tmp = pd.read_csv('data/answer_train.csv')
df_label = pd.DataFrame({'label': train_tmp.label.value_counts(normalize=True).index.tolist(),
                         'label_n': [i for i in range(train_tmp.label.nunique())]})

train_df = train_df.sample(frac=1., random_state=1024)
# eval_df = train_df[len(train_df)*0.9:]
# train_df = train_df[:54000]
train_df, eval_df = train_test_split(train_df, test_size=0.1)
model_args = ClassificationArgs()

model_args.max_seq_length = 150
model_args.train_batch_size = 32
model_args.num_train_epochs = 5
model_args.fp16 = False
model_args.evaluate_during_training = True
model_args.overwrite_output_dir = True

model = ClassificationModel("bert",
                            "hfl/chinese-roberta-wwm-ext",
                            # "hfl/chinese-roberta-wwm-ext-large",
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
sub.head(10)
np.save('result/chinese-roberta-wwm-ext.npy', raw_outputs)
sub.to_csv('result/baseline_20201117_.csv', index=False)

