import numpy as np

from nn.interface import IOperator
from nn.loss.abstract import ILoss


class MSELoss(ILoss):

    def __init__(self):
        pass

    def __repr__(self):
        print(self.__str__())

    def __str__(self):
        return "<Mean Square Error Loss>"

    def gradient(self, arg1, arg2):
        # n = arg1.shape[0]
        return arg1 - arg2, -(arg1 - arg2)

    def metric(self, arg1, arg2):
        # np.mean(a, axis=),axis不设置值，表示对a中的所有数求平均值，返回一个实数
        return 0.5 * np.mean(np.sum(np.power(arg1 - arg2, 2), axis=1))
        # return np.sum(np.square(arg1 - arg2))/arg1.shape[0]


if __name__ == '__main__':
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[3, 4], [2, 7]])
    c = np.array([[8, 9], [1, 7]])
    mse = MSELoss()
    print(mse.metric(a, c))
    d = np.array([1, 2, 3, 4])
    e = [[1, 2, 3, 4], [2, 3, 4, 5]]
    print(np.mean(e, axis=0))
