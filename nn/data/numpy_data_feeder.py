from numpy import ndarray
from nn.data.interface import IDataFeeder
import numpy as np


class NumpyDataFeeder(IDataFeeder):

    def __init__(self, x: ndarray, y: ndarray, batch_size: int):
        self.__iter_x: ndarray = x
        self.__iter_y: ndarray = y
        self.__batch_size = min(batch_size, len(x))
        self.__batches = len(x) // self.__batch_size
        self.__iter = 0

    @property
    def position(self):
        return self.__iter

    @property
    def length(self):
        return self.__batches

    @property
    def batch_size(self):
        return self.__batch_size

    def iter_x(self):
        return self.__iter_x

    def set_iter_x(self, iter_x):
        self.__iter_x = iter_x

    def iter_y(self):
        return self.__iter_y

    def set_iter_y(self, iter_y):
        self.__iter_y = iter_y

    def __iter__(self):
        for self.__iter in range(self.__batches):
            start = self.__iter * self.__batch_size % (len(self.__iter_x) - self.__batch_size + 1)
            end = start + self.__batch_size
            part_x = self.__iter_x[start:end]
            part_y = self.__iter_y[start:end]
            self.__iter += 1
            yield part_x, part_y

    def __repr__(self):
        print(self.__str__())

    def __str__(self):
        return "<Numpy 01 iterator, current batch: {}, total: {}.>".format(self.__iter, self.__batches)


if __name__ == '__main__':
    x = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    indices = np.arange(10)
    np.random.shuffle(indices)
    x = x[indices]
    print(x)
