import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.data.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


root_path = './dataset/'
data_path = 'jeju_specialty.csv'

scaler = StandardScaler()
df_raw = pd.read_csv(os.path.join(root_path,
                                    data_path))
df_raw
df_raw = df_raw.iloc[:, 1:]
# df_raw.to_csv("./dataset/jeju_specialty.csv", index=False)

'''
df_raw.columns: [time_col_name, ...(other features), target feature]
'''
cols = list(df_raw.columns)
#cols.remove(target) if target
#cols.remove(time_col_name)
#df_raw = df_raw[[time_col_name] + cols + [target]]

num_train = int(len(df_raw) * train_split)
num_test = int(len(df_raw) * test_split)
num_vali = len(df_raw) - num_train - num_test
border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
border2s = [num_train, num_train + num_vali, len(df_raw)]
border1 = border1s[set_type]
border2 = border2s[set_type]

if features == 'M' or features == 'MS':
    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data]
elif features == 'S':
    df_data = df_raw[[target]]

if scale:
    train_data = df_data[border1s[0]:border2s[0]]
    scaler.fit(train_data.values)
    data = scaler.transform(df_data.values)
else:
    data = df_data.values

df_stamp = df_raw[[time_col_name]][border1:border2]
df_stamp[time_col_name] = pd.to_datetime(df_stamp[time_col_name])
if timeenc == 0:
    df_stamp['year'] = df_stamp[time_col_name].apply(lambda row: row.year, 1)
    df_stamp['month'] = df_stamp[time_col_name].apply(lambda row: row.month, 1)
    df_stamp['day'] = df_stamp[time_col_name].apply(lambda row: row.day, 1)
    df_stamp['weekday'] = df_stamp[time_col_name].apply(lambda row: row.weekday(), 1)
    data_stamp = df_stamp.drop([time_col_name], axis=1).values
elif timeenc == 1:
    data_stamp = time_features(pd.to_datetime(df_stamp[time_col_name].values), freq=freq)
    data_stamp = data_stamp.transpose(1, 0)

data_x = data[border1:border2]
data_y = data[border1:border2]
data_stamp = data_stamp