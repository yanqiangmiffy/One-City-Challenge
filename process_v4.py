import warnings

warnings.simplefilter('ignore')
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re

train = pd.read_csv('data/answer_train.csv')
test = pd.read_csv('data/submit_example_test1.csv')
df_label = pd.DataFrame({'label': train.label.value_counts(normalize=True).index.tolist(),
                         'label_n': [i for i in range(train.label.nunique())]})

train = train.merge(df_label, on='label', how='left')


def clean_text(x):
    text = x.replace('train/', '').replace('.xls', '').replace('.csv',
                                                               '').replace('_', ' ').replace('test1/', '')
    return text


train['text'] = train['filename'].apply(lambda x: clean_text(x))
test['text'] = test['filename'].apply(lambda x: clean_text(x))


def get_file_content(filename):
    table_path = 'data/' + filename
    r1 = '[0-9’!"#$%&\'()）*+-./:;,<=>?@。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    if file_name.endswith('xls'):
        try:
            with open(table_path, 'r', encoding='utf-8') as f:
                text = "".join(f.read().split())
                text = re.sub(r1, '', text)
                print("读取xls方式[open]成功", table_path)
                return text[:300]
        except UnicodeDecodeError as e:
            try:
                df = pd.read_excel(table_path)
                print("读取xls方式[read_excel]成功", table_path)
                if len(df) == 0:
                    data = pd.DataFrame()
                    tmp_xls = pd.ExcelFile(table_path)
                    sheet_names = tmp_xls.sheet_names
                    for name in sheet_names:
                        d = tmp_xls.parse(name)
                        data = pd.concat([data, d])
                    text = data.to_string()
                    text = "".join(text.split())
                    text = re.sub(r1, '', text)
                    text = text.replace('NaN', '').replace('\n', '')
                    # print(text)
                    return text[:300]
                else:
                    text = df.to_string()
                    text = "".join(text.split())
                    text = re.sub(r1, '', text)
                    text = text.replace('NaN', '').replace('\n', '')

                    # print(text)
                    return text[:300]

            except Exception as e:
                try:
                    df = pd.read_html(table_path)
                    print("读取xls方式[read_html]成功", table_path)
                    text = df.to_string()
                    text = "".join(text.split())
                    text = re.sub(r1, '', text)
                    text = text.replace('NaN', '').replace('\n', '')

                    # print(text)
                    return text[:300]
                except Exception as e:
                    print(e)
                    print("读取xls失败", table_path)
                    return ''
    elif file_name.endswith('csv'):
        try:
            df = pd.read_csv(table_path, error_bad_lines=False, warn_bad_lines=False, lineterminator='\n')
            text = df.to_string()
            text = "".join(text.split())
            text = re.sub(r1, '', text)
            text = text.replace('NaN', '').replace('\n', '')
            return text[:300]
        except Exception as e:
            return ''


content = []
for file_name in train['filename']:
    print("======================正在处理{}============".format(file_name))
    content.append(get_file_content(file_name))

train['content'] = content
train['text'] = train['text'].astype(str) + ' ' + train['content'].astype(str)

content = []
for file_name in test['filename']:
    print("======================正在处理{}============".format(file_name))
    content.append(get_file_content(file_name))
    print("\n")
test['content'] = content
test['text'] = test['text'].astype(str) + ' ' + test['content'].astype(str)


train_df = train[['text', 'label_n']]
train_df.columns = ['text', 'label']
test_df = test[['text', 'label']]

# print(train_df)
# print(train_df.label.unique(),train_df.label.nunique())
# print(train_df['label'].value_counts())
# print(test)

train_df.to_csv('data/train_set.csv', index=None)
test_df.to_csv('data/test_set.csv', index=None)

