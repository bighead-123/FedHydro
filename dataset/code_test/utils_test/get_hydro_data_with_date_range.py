import numpy as np
import pandas as pd
from dataset.series_data.utils.camels_operate import CamelsOperate
from utils.constants import path


class GetHydroDataWithDate:
    def __init__(self, basin_id, sequence_length, date_range):
        self.__basin_id = basin_id
        self.__date_range = date_range
        self.__sequence_length = sequence_length
        self.__path = path
        self.__ds_train = None
        self.__ds_val = None
        self.__ds_test = None
        self.__x = None
        self.__y = None

    def get_data(self):
        file_path = self.__path
        basin = self.__basin_id
        sequence_length = self.__sequence_length
        train_start_date = self.__date_range['train_date']['start_date']
        train_end_date = self.__date_range['train_date']['end_date']
        ds_train = CamelsOperate(file_path=file_path, basin=basin, seq_length=sequence_length, period="train",
                                 dates=[train_start_date, train_end_date])
        self.__ds_train = ds_train
        self.__x, self.__y = ds_train.get_discharge()
        print("数据长度：", ds_train.__len__())
        print("ds_train[0]，长度", ds_train[0], len(ds_train[0][0]))
        train_x = np.asarray(ds_train.x)
        train_y = np.asarray(ds_train.y)

        means = ds_train.get_means()
        stds = ds_train.get_stds()
        val_start_date = self.__date_range['val_date']['start_date']
        val_end_date = self.__date_range['val_date']['end_date']
        ds_val = CamelsOperate(file_path=file_path, basin=basin, seq_length=sequence_length, period="eval",
                                dates=[val_start_date, val_end_date],
                                means=means, stds=stds)

        self.__ds_val = ds_val
        val_x = np.asarray(ds_val.x)
        val_y = np.asarray(ds_val.y)

        means = ds_train.get_means()
        stds = ds_train.get_stds()
        test_start_date = self.__date_range['test_date']['start_date']
        test_end_date = self.__date_range['test_date']['end_date']
        ds_test = CamelsOperate(file_path=file_path, basin=basin, seq_length=sequence_length, period="eval",
                               dates=[test_start_date, test_end_date],
                               means=means, stds=stds)

        self.__ds_test = ds_test
        test_x = np.asarray(ds_test.x)
        test_y = np.asarray(ds_test.y)

        return train_x, train_y, val_x, val_y, test_x, test_y

    def get_ds_train(self):
        return self.__ds_train

    def get_ds_val(self):
        return self.__ds_val

    def get_ds_test(self):
        return self.__ds_test

    def ds_test_renormalize_discharge(self, Qobs):
        return self.__ds_test.reshape_discharge(Qobs)

    def get_discharge_data(self):
        return self.__x, self.__y


if __name__=='__main__':
    getdata = GetHydroData("01030500", 30)
    getdata.get_data()