import multiprocessing as mp
import re
import warnings
import pandas as pd
from tqdm import tqdm
"""
基本思路： 对表格文本数据进行提取，文本由三部分组成：文件名+表格名字+表头列名

文件名：直接用文件名，利用baseline代码可以达到一个不错的基线成绩，0.977+
表格名字：有的xls内容包含多个表格，可以利用这部分表格的名字提供额外的信息，后来发现这部分文本缺陷比较大，主要是test，少部分有中文文本，另外就是为空
表头列名：这部分信息作用比较大，因为这些数据其实从政府职能网站爬去下来，数据比较脏，但是网站的html或者格局是固定的，所以表头可以提供比较有效的信息

三段文本拼接方式不同，效果也会不同
"""


def clean_text(x):
    text = x.replace('train/', '').replace('.xls', '').replace('.csv',
                                                               '').replace('_', ' ').replace('test1/', '')
    return text


def process_text(text):
    r1 = '[a-zA-Z0-9’!"#$%&\'()）*+-./:;,<=>?@。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(r1, '', text)
    text = text.replace('NaN', '').replace('\n', '')
    text = text.replace("\\", "")
    # print(text)
    text = "".join(text.split())
    return text


def get_file_content_v1(filename):
    """
    直接将表格内容进行拼接
    """
    table_path = 'data/' + filename
    r1 = '[0-9’!"#$%&\'()）*+-./:;,<=>?@。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    if filename.endswith('xls'):
        try:
            with open(table_path, 'r', encoding='utf-8') as f:
                text = "".join(f.read().split())
                text = re.sub(r1, '', text)
                print("读取xls方式[open]成功", table_path)
                text=process_text(text)
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
                    text=process_text(text)
                    # print(text)
                    return text[:300]
                else:
                    text = df.to_string()
                    text = "".join(text.split())
                    text = re.sub(r1, '', text)
                    text = text.replace('NaN', '').replace('\n', '')
                    text=process_text(text)

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
                    text=process_text(text)

                    # print(text)
                    return text[:300]
                except Exception as e:
                    print(e)
                    print("读取xls失败", table_path)
                    return ''
    elif filename.endswith('csv'):
        try:
            df = pd.read_csv(table_path, error_bad_lines=False, warn_bad_lines=False, lineterminator='\n')
            text = df.to_string()
            text = "".join(text.split())
            text = re.sub(r1, '', text)
            text = text.replace('NaN', '').replace('\n', '')
            text=process_text(text)
            return text[:300]
        except Exception as e:
            return ''



if __name__ == '__main__':
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
    test = pd.read_csv('data/submit_example_test2.csv')
    test['text']=test['filename'].apply(lambda x:process_text(x))
    print(test.head())
    contents=[]
    for name in tqdm(test['filename']):
        contents.append(get_file_content_v1(name))

    test['text']=test['filename'].apply(lambda x:process_text(x))
    print(test.head())
    test['content'] = contents
    # test['content'] = test['filename'].parallel_apply(get_file_content).values
    test['text'] = test['text'].astype(str) + ' ' + test['content'].astype(str)

    test_df = test[['text', 'label']]
    test_df.to_csv('data/test_set_v3_dan.csv', index=None)

