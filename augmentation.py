import pandas as pd
from tqdm import tqdm


def aug_df(df=None, selected=[15, 16, 17, 18, 19], text_len=100):
    """
    工业            8542
    文化休闲          7408
    教育科技          7058
    医疗卫生          5750
    文秘行政          5439
    生态环境          4651
    城乡建设          3722
    农业畜牧业         2703
    经济管理          2516
    交通运输          2250
    政法监察          2159
    财税金融          1784
    劳动人事          1759
    旅游服务          1539
    资源能源          1209
    商业贸易           652
    气象水文测绘地震地理     375
    民政社区           349
    信息产业           108
    外交外事            27


                 label  label_n
    0           工业        0
    1         文化休闲        1
    2         教育科技        2
    3         医疗卫生        3
    4         文秘行政        4
    5         生态环境        5
    6         城乡建设        6
    7        农业畜牧业        7
    8         经济管理        8
    9         交通运输        9
    10        政法监察       10
    11        财税金融       11
    12        劳动人事       12
    13        旅游服务       13
    14        资源能源       14
    15        商业贸易       15
    16  气象水文测绘地震地理       16
    17        民政社区       17
    18        信息产业       18
    19        外交外事       19
    :param df:
    :return:
    """

    # 外交
    # tmp0 = df[df.label == 15]
    # tmp1 = df[df.label == 16]
    # tmp2 = df[df.label == 17]
    # tmp3 = df[df.label == 18]
    # tmp4 = df[df.label == 19]
    data_list = []
    data = df[df.label.isin(selected)]
    for index, row in tqdm(data.iterrows()):
        # print(index,row)
        text=row.text
        if len(text) > 300:
            text=text[:1500]
            # print("====" * 2000)
            filename = text.split(' ')[0]
            candidate_text = text.replace(filename, '')
            # print([filename, candidate_text])
            for text_index in range(0, len(candidate_text), text_len):
                # print(text_index)
                data_list.append([file_name+' '+candidate_text[text_index:text_index +70], row.label])

    result = pd.DataFrame(data_list, columns=['text', 'label'])
    result = pd.concat([df, result], axis=0)
    return result

