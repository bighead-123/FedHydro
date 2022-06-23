import numpy as np
import os
import hashlib
from dataset.femnist import IFedDataset
from utils.constants import path


class HydroDataSetFedSRP(IFedDataset):

    def set_node_id(self, node_id: int) -> None:
        pass

    def __init__(self, check_sum=None):

        # 路径写成这样是因为是在主程序调用当前类
        self.path = './dataset/series_data/'
        # self.path = path
        super().__init__(check_sum)

    def __repr__(self):
        return '<FedSRP dataset.>'

    # 获取数据文件校验和
    def check_sum(self) -> str:

        # os.path.join() 路径拼接
        if not (os.path.exists(os.path.join(self.path, 'train_unit01_8basins_t30_x.txt')) and
                os.path.exists(os.path.join(self.path, 'train_unit01_8basins_t30_y.txt')) and
                os.path.exists(os.path.join(self.path, 'test_unit01_8basins_t30_x.txt')) and
                os.path.exists(os.path.join(self.path, 'test_unit01_8basins_t30_y.txt'))):
            return ''

        sum = hashlib.md5()  # 获取一个md5加密算法对象
        with open(os.path.join(self.path, 'train_unit01_8basins_t30_x.txt'), 'rb') as file:
            sum.update(file.read())  # 制定需要加密的字符串
        with open(os.path.join(self.path, 'train_unit01_8basins_t30_y.txt'), 'rb') as file:
            sum.update(file.read())
        with open(os.path.join(self.path, 'test_unit01_8basins_t30_x.txt'), 'rb') as file:
            sum.update(file.read())
        with open(os.path.join(self.path, 'test_unit01_8basins_t30_y.txt'), 'rb') as file:
            sum.update(file.read())
        return sum.hexdigest()  # 获取加密后的16进制字符串

    def extract_files(self) -> list:
        return [
            os.path.join(self.path, 'train_unit01_8basins_t30_x.txt'),
            os.path.join(self.path, 'train_unit01_8basins_t30_y.txt'),
            os.path.join(self.path, 'test_unit01_8basins_t30_x.txt'),
            os.path.join(self.path, 'test_unit01_8basins_t30_y.txt')
        ]

    def estimate_size(self) -> int:
        return 87988326  # 约90M

    def __load_core(self, kind='train'):
        """Load hydro 01 from `path`"""
        x_path = os.path.join(self.path, '%s_unit01_8basins_t30_x.txt' % kind)
        y_path = os.path.join(self.path, '%s_unit01_8basins_t30_y.txt' % kind)
        # 注意, time_step和generate_data中的对应起来
        time_step = 30
        x = np.loadtxt(x_path, delimiter=' ').reshape((-1, time_step, 5))
        labels = np.loadtxt(y_path, delimiter=' ').reshape((-1, 1))
        return x, labels

    def load(self):
        train_x, train_y = self.__load_core(kind='train')
        test_x, test_y = self.__load_core(kind='test')
        return train_x, np.reshape(train_y, (-1)), test_x, np.reshape(test_y, (-1))


if __name__ == '__main__':
    # basin_ids = {'01013500', '01022500', '01030500'}
    # date_range = {
    #     'train_date': {
    #         'start_date': pd.to_datetime("1980-10-01", format="%Y-%m-%d"),
    #         'end_date': pd.to_datetime("1995-09-30", format="%Y-%m-%d")
    #     },
    #
    #     'test_date': {
    #         'start_date': pd.to_datetime("1995-10-01", format="%Y-%m-%d"),
    #         'end_date': pd.to_datetime("2000-09-30", format="%Y-%m-%d")
    #     }
    # }
    hydroDataSet = HydroDataSetFedSRP()
    print(hydroDataSet.check_sum())
    print(hydroDataSet.extract_files())
    train_x, train_y, test_x, test_y = hydroDataSet.load()
    print(train_x)
    print(train_y)
