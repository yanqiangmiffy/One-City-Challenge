import pandas as pd
import os
import shutil
from tqdm import tqdm
print(os.listdir('data'))

if not os.path.exists('data/cates'):
    os.mkdir('data/cates')

source_dir='data/'
target_dir='data/cates/'

label_df=pd.read_csv('data/answer_train.csv')

for label in label_df['label'].unique():
    if not os.path.exists(target_dir+label):
        os.mkdir(target_dir+label)


for index,row in tqdm(label_df.iterrows()):
    source_file=source_dir+row['filename']
    target_file=target_dir+row['label']+'/'+row['filename'].replace('train/','')
    shutil.copyfile(source_file, target_file)
