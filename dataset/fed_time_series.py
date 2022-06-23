import struct
import numpy as np
import os
import hashlib

from dataset.femnist import IFedDataset


class TimeSeries(IFedDataset):

    def set_node_id(self, node_id: int) -> None:
        pass

    def __init__(self, check_sum=None):
        # 路径写成这样是因为是在主程序调用当前类
        # self.path = './dataset/series_data/'
        self.path = 'D:/河海大学/研究课题/研究课题/实验相关/PSGD/PSGD-0515/Parallel-SGD-0515/dataset/series_data'
        super().__init__(check_sum)

    def __repr__(self):
        return '<TEST prediction dataset.>'

    # 获取数据文件校验和
    def check_sum(self) -> str:

        # os.path.join() 路径拼接
        if not (os.path.exists(os.path.join(self.path, 'train_x.txt')) and
                os.path.exists(os.path.join(self.path, 'train_y.txt')) and
                os.path.exists(os.path.join(self.path, 'test_x.txt')) and
                os.path.exists(os.path.join(self.path, 'test_y.txt'))):
            return ''

        sum = hashlib.md5()   # 获取一个md5加密算法对象
        with open(os.path.join(self.path, 'train_x.txt'), 'rb') as file:
            sum.update(file.read())  # 制定需要加密的字符串
        with open(os.path.join(self.path, 'train_y.txt'), 'rb') as file:
            sum.update(file.read())
        with open(os.path.join(self.path, 'test_x.txt'), 'rb') as file:
            sum.update(file.read())
        with open(os.path.join(self.path, 'test_y.txt'), 'rb') as file:
            sum.update(file.read())
        return sum.hexdigest()  # 获取加密后的16进制字符串

    def extract_files(self) -> list:
        return [
            os.path.join(self.path, 'train_x.txt'),
            os.path.join(self.path, 'train_y.txt'),
            os.path.join(self.path, 'test_x.txt'),
            os.path.join(self.path, 'test_y.txt')
            ]

    def estimate_size(self) -> int:
        return 40960  # 40KB

    def __load_core(self, kind='train'):
        """Load hydro 01 from `path`"""
        x_path = os.path.join(self.path, '%s_x.txt' % kind)
        y_path = os.path.join(self.path, '%s_y.txt' % kind)

        x = np.loadtxt(x_path, delimiter=' ').reshape((-1, 1, 5))
        labels = np.loadtxt(y_path, delimiter=' ').reshape((-1, 1))
        return x, labels

    def load(self):
        train_x, train_y = self.__load_core(kind='train')
        test_x, test_y = self.__load_core(kind='test')
        return train_x, np.reshape(train_y, (-1)), test_x, np.reshape(test_y, (-1))


if __name__ == '__main__':
    fed = TimeSeries()
    fed.set_node_id(1)
    x, y, x_t, y_t = fed.load()
    print(fed.check_sum())
    print(fed.extract_files())
    print(x)
    print(y)
