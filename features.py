#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: features.py 
@time: 2020/11/23 11:26 下午
@description:
"""
import numpy as np
import time
import os
import re
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

tqdm.pandas()
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
    text = x.replace('train/', '').replace('.xls', '').replace('.csv', '').replace('_', ' ').replace('test1/', '')
    return text


def process_text_colsheet(text):
    r1 = '[a-zA-Z0-9’!"#$%&\'()）*+-./:;,<=>?@。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(r1, '', text)
    text = text.replace('NaN', '').replace('\n', '')
    text = text.replace("\\", "")
    # print(text)
    text = "".join(text.split())
    return text


def process_text_content(text):
    r1 = '[a-zA-Z0-9’!"#$%&\'()）*+-./:;,<=>?@。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(r1, '', text)
    text = text.replace('NaN', '').replace('\n', '').replace('\r', '')
    text = text.replace("\\", "")
    # print(text)
    text = "".join(text.split())
    return text


def trim_html(html):
    dr = re.compile(r'<[^>]+>', re.S)
    dd = dr.sub('', html)
    return dd


def get_features(result):
    filename = result[0]
    sheet_names = result[1]
    column_names = result[2]
    table_contents = result[3]
    filename_text = clean_text(filename)

    sheet_names_text = process_text_colsheet("".join(sheet_names))
    sheet_names_text_len = len(sheet_names_text)
    sheet_names_nums = len(sheet_names_text)
    sheet_names_nodhup_nums = len(set(sheet_names_text))

    column_names_text = process_text_colsheet("".join(column_names))
    column_names_text_len = len(column_names_text)
    column_nums = len(column_names)

    table_nums = len(table_contents)
    num_records = 0
    table_content_text = ''
    for table in table_contents:
        num_records += len(table_contents)
        table_content_text += table.to_string(header=False, show_dimensions=False, index=False, index_names=False,
                                              sparsify=False)
    table_text_len = len(table_content_text)
    table_text_len_nohtml = len(trim_html(table_content_text))
    processed_text = process_text_content(table_content_text)
    # print(processed_text)
    table_text_len_nonumaz = len(processed_text)
    return filename, filename_text, sheet_names_text, column_names_text, sheet_names_text_len, sheet_names_nums, sheet_names_nodhup_nums, \
           column_names_text_len, column_nums, table_nums, table_text_len,\
           table_text_len_nohtml, table_text_len_nonumaz, processed_text[:100]


def get_file_content(filename):
    table_path = 'data/' + filename

    if filename.endswith('xls'):
        try:
            sheet_names = []
            col_names = []
            table_contents = []

            tmp_xls = pd.ExcelFile(table_path)
            sheet_names = tmp_xls.sheet_names
            sheet_names.extend(sheet_names)
            for name in sheet_names:
                d = tmp_xls.parse(name)
                try:
                    col_names.extend(d.columns.tolist())
                except Exception as e:
                    col_names.extend([])
                table_contents.append(d)
            print("处理成功", table_path)
            res = (filename, sheet_names, col_names, table_contents)
            return get_features(res)
        except Exception as e:
            try:
                sheet_names = []
                col_names = []
                table_contents = []
                data = pd.read_csv(table_path, error_bad_lines=False, warn_bad_lines=False, lineterminator='\n')
                try:
                    col_names.extend(data.columns.tolist())
                except Exception as e:
                    col_names.extend([])
                table_contents.append(data)
                res = (filename, sheet_names, col_names, table_contents)
                return get_features(res)
            except Exception as e:
                sheet_names = []
                col_names = []
                table_contents = []
                res = (filename, sheet_names, col_names, table_contents)
                return get_features(res)
    elif filename.endswith('csv'):
        try:
            sheet_names = []
            col_names = []
            table_contents = []
            data = pd.read_csv(table_path, error_bad_lines=False, warn_bad_lines=False, lineterminator='\n')
            try:
                col_names.extend(data.columns.tolist())
            except Exception as e:
                col_names.extend([])
            table_contents.append(data)
            res = (filename, sheet_names, col_names, table_contents)
            return get_features(res)
        except Exception as e:
            sheet_names = []
            col_names = []
            table_contents = []
            res = (filename, sheet_names, col_names, table_contents)
            return get_features(res)


if __name__ == '__main__':
    features = ['filename', 'filename_text', 'sheet_names_text', 'column_names_text', 'sheet_names_text_len',
                'sheet_names_nums', 'sheet_names_nodhup_nums',
                'column_names_text_len', 'column_nums', 'table_nums', 'table_text_len', 'table_text_len_nohtml',
                'table_text_len_nonumaz', 'processed_text']
    pool = Pool(os.cpu_count() - 1)
    with open('data/test_content.csv', 'w', encoding='utf-8') as f:
        f.write(",".join(features) + '\n')
        results = pool.map(get_file_content, test['filename'])
        print(len(results))
        #for result in pool.imap(get_file_content, test['filename']):
        for result in results:    
            line=("{}," * len(features)).format(*result).rstrip(',') + "\n"
            f.write(line)
        del results

    pool.close()
    pool.join()

    #pool = Pool(os.cpu_count() - 1)
    with open('data/train_content.csv', 'w', encoding='utf-8') as f:
        f.write(",".join(features) + '\n')
        data = np.split(train, range(10000, len(train), 10000)) 
        for tmp_train in data:
            pool = Pool(os.cpu_count() - 1)
            results = pool.map(get_file_content, tmp_train['filename'])
            for result in results:
                line=("{}," * len(features)).format(*result).rstrip(',') + "\n"
                f.write(line)
                pool.close()
                pool.join()
            time.sleep(1)
            del results
            print("="*50)

    df=pd.read_csv('data/test_content.csv')
    test=pd.read_csv('data/submit_example_test1.csv')
    data=pd.merge(test,df,how="left",on="filename")
    data.to_csv('data/test_content.csv',index=None)

    df=pd.read_csv('data/train_content.csv')
    label=pd.read_csv('data/answer_train.csv')
    data=pd.merge(label,df,how="left",on="filename")
    data.to_csv("data/train_content.csv",index=None)
