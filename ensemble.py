import numpy as np
import pandas as pd



train_tmp = pd.read_csv('data/answer_train.csv')
df_label = pd.DataFrame({'label': train_tmp.label.value_counts(normalize=True).index.tolist(),
                         'label_n': [i for i in range(train_tmp.label.nunique())]})



xlnet=np.load('result/03_chinese-xlnet-base_p3.npy')
albert=np.load('result/albert_chinese_base.npy')
roberta_p3=np.load('result/chinese-roberta-wwm-ext.npy')
roberta_p6=np.load('result/chinese-roberta-wwm-ext_p6.npy')
elec=np.load('result/04_chinese-electra-base-discriminator.npy')
longformer=np.load('result/05_longformer-chinese-base-4096.npy')
elec_gen=np.load('result/06_chinese-electra-base-gen.npy')
roberta_base=np.load('result/07_roberta_base.npy')

pred=xlnet+roberta_p3+roberta_p6+elec+longformer+roberta_base
#print(xlnet)
#print(pred)
#pred=elec
predictions=np.argmax(pred,axis=1)
sub = pd.read_csv('data/submit_example_test1.csv')[['filename']]
sub['label_n'] = predictions
sub = pd.merge(sub, df_label, on='label_n', how='left')
sub.drop(['label_n'], axis=1, inplace=True)

print(sub.shape)
print(sub.head(10))




sub.to_csv('result/ense.csv', index=False)
