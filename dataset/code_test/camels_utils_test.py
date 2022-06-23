import pandas as pd
from dataset.series_data.utils.camels_operate import CamelsOperate
import numpy as np
date_range = {
        'train_date': {
            'start_date': pd.to_datetime("1980-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("1995-09-30", format="%Y-%m-%d")
        },

        'test_date': {
            'start_date': pd.to_datetime("1995-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2000-09-30", format="%Y-%m-%d")
        }
    }
train_start_date = date_range['train_date']['start_date']
train_end_date = date_range['train_date']['end_date']
basin_id = '01030500'
# basin_id = '01030500'
time_step = 30
file_path = 'D:/河海大学/研究课题/水文预报/课题实验/Parallel-SGD/dataset/series_data/'
ds_train = CamelsOperate(file_path, basin=basin_id, seq_length=time_step, period="train",
                         dates=[train_start_date, train_end_date])
train_x = ds_train.x
train_y = ds_train.y

with open('D:\\河海大学\\研究课题\\水文预报\\课题实验\\Parallel-SGD\\dataset\\train_x.txt',
          'w') as outfile:
    for slice in train_x:
        np.savetxt(outfile, slice, fmt='%f', delimiter=' ')

with open('D:\\河海大学\\研究课题\\水文预报\\课题实验\\Parallel-SGD\\dataset\\train_y.txt',
          'w') as outfile:
    for slice in train_y:
        np.savetxt(outfile, slice, fmt='%f', delimiter=' ')