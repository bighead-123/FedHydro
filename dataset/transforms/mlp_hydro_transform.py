import numpy as np
from dataset.transforms.abstract import AbsTransformer


class MlpDataTransform(AbsTransformer):
    def __init__(self, batch_size):
        super().__init__()
        self.__batch_size = batch_size

    def __repr__(self):
        return "<Reshape time series 01>"

    @property
    def params(self):
        return ()

    def run(self, train_x, train_y, test_x, test_y) -> tuple:
        # get total sample count
        n = train_y.shape[0]
        # get real n
        # 保证总数据量可以被batch_size整数，以便数据划分
        n = n - n % self.__batch_size
        train_x = train_x[:n]
        train_y = train_y[:n]
        input_dim = train_x.shape[1] * train_x.shape[2]
        # reshape 01, [-1,30, 5]  --->  [-1, 30*5]
        train_x = np.reshape(train_x, newshape=(train_x.shape[0], input_dim))
        test_x = np.reshape(test_x, newshape=(test_x.shape[0], input_dim))
        return train_x, np.reshape(train_y, (-1, 1)), test_x, np.reshape(test_y, (-1, 1))