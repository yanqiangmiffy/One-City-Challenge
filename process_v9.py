import warnings
import pandas as pd
import re
import numpy as np
from multiprocessing import Pool
from pandarallel import pandarallel
import logging

pandarallel.initialize(progress_bar=True, nb_workers=8)

train = pd.read_csv('data/answer_train.csv')
test = pd.read_csv('data/submit_example_test1.csv')
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


def clean_text(x):
    text = x.replace('train/', '').replace('.xls', '').replace('.csv',
                                                               '').replace('_', ' ').replace('test1/', '')
    return text


train['file'] = train['filename'].apply(lambda x: clean_text(x))
test['file'] = test['filename'].apply(lambda x: clean_text(x))


def get_file_content(filename):
    table_path = 'data/' + filename

    if filename.endswith('xls'):
        try:

            data = pd.DataFrame()
            tmp_xls = pd.ExcelFile(table_path)
            sheet_names = tmp_xls.sheet_names
            # print("表格名字：",sheet_names," ".join(sheet_names))
            sheet_name_text = " ".join(sheet_names)
            col_names = []
            for name in sheet_names:
                d = tmp_xls.parse(name)
                col_names.extend(d.columns.tolist())
                data = pd.concat([data, d])
            # print("表头名字", col_names)
            col_name_text = " ".join(col_names)
            table_content_text = data.to_string(header=False, show_dimensions=False, index=False, index_names=False,
                                                sparsify=False)
            logging.info("处理成功", table_path)

            return sheet_name_text, col_name_text, table_content_text[:300]
        except Exception as e:
            # print(e, table_path)
            try:

                data = pd.read_csv(table_path, error_bad_lines=False, warn_bad_lines=False, lineterminator='\n')
                sheet_name_text = ''
                col_name_text = " ".join(data.columns.tolist())
                table_content_text = data.to_string(header=False, show_dimensions=False, index=False, index_names=False,
                                                    sparsify=False)
                logging.info("处理成功", table_path)

                return sheet_name_text, col_name_text, table_content_text[:300]
            except Exception as e:
                logging.error(e, table_path)
                sheet_name_text = ''
                col_name_text = ''
                table_content_text = ''
                return sheet_name_text, col_name_text, table_content_text

    if filename.endswith('csv'):
        try:
            df = pd.read_csv(table_path, error_bad_lines=False, warn_bad_lines=False, lineterminator='\n')
            sheet_name_text = ''
            col_name_text = " ".join(df.columns.tolist())
            table_content_text = df.to_string(header=False, show_dimensions=False, index=False, index_names=False,
                                              sparsify=False)
            logging.info("处理成功", table_path)

            return sheet_name_text, col_name_text, table_content_text[:300]
        except Exception as e:
            logging.error(e, table_path)
            sheet_name_text = ''
            col_name_text = ''
            table_content_text = ''
            return sheet_name_text, col_name_text, table_content_text


def process_text(text):
    r1 = '[a-zA-Z0-9’!"#$%&\'()）*+-./:;,<=>?@。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(r1, '', text)
    text = text.replace('NaN', '').replace('\n', '')
    print(text)
    return text[:500]


if __name__ == '__main__':
    res_parallel = test['filename'].parallel_apply(get_file_content)
    res_parallel = pd.DataFrame([list(r) for r in res_parallel.values.tolist()],
                                columns=['sheet_name_text', 'col_name_text', 'table_content_text'])
    test = pd.concat([test, res_parallel], axis=1)
    test['text'] = test['file'].str.cat(test[['sheet_name_text', 'col_name_text', 'table_content_text']], sep=' ')
    print(test.head())
    print("test processing done")

    res_parallel = train['filename'].parallel_apply(get_file_content)
    res_parallel = pd.DataFrame([list(r) for r in res_parallel.values.tolist()],
                                columns=['sheet_name_text', 'col_name_text', 'table_content_text'])
    train = pd.concat([train, res_parallel], axis=1)
    train['text'] = train['file'].str.cat(train[['sheet_name_text', 'col_name_text', 'table_content_text']], sep=' ')
    print("train processing done")

    # print(train_df)
    # print(train_df.label.unique(),train_df.label.nunique())
    # print(train_df['label'].value_counts())
    # print(test)
    train['text'] = train['text'].apply(lambda x: process_text(x))
    test['text'] = test['text'].apply(lambda x: process_text(x))

    train['label'] = train['label'].map(label_index)
    test['label'] = test['label'].map(label_index)

    train_df = train[['filename', 'label', 'text', 'sheet_name_text', 'col_name_text', 'table_content_text']]
    test_df = test[['filename', 'label', 'text', 'sheet_name_text', 'col_name_text', 'table_content_text']]

    train_df.to_csv('data/train_set_v9.csv', index=None)
    test_df.to_csv('data/test_set_v9.csv', index=None)

