import os
import numpy as np
from dataset.code_test.fed_meta_ref.MAML.data.extract_node_data import SingleNodeData


class HydroDataMeta:
    def __init__(self, path):
        self.__path = path

    def load_core(self, path, unit_code, node_count):
        """Load hydro 01 from `path`"""
        """对于unit01，train_6basins_x/y，第一个流域为01030500"""
        x_path = str(unit_code) + "\\" + "train_unit"+str(unit_code)+"_"+str(node_count)+"basins_t30_x.txt"
        y_path = str(unit_code) + "\\" + "train_unit"+str(unit_code)+"_"+str(node_count)+"basins_t30_y.txt"
        x_path = os.path.join(path,  x_path)
        y_path = os.path.join(path, y_path)
        # 注意, time_step和generate_data中的对应起来
        time_step = 30
        x = np.loadtxt(x_path, delimiter=' ').reshape((-1, time_step, 5))
        labels = np.loadtxt(y_path, delimiter=' ').reshape((-1, 1))
        return x, labels

    def load_core_test(self, path, unit_code, node_count):
        """Load hydro 01 from `path`"""
        x_path = str(unit_code) + "\\" + "test_unit" + str(unit_code) + "_" + str(
            node_count) + "basins_t30_x.txt"
        y_path = str(unit_code) + "\\" + "test_unit" + str(unit_code) + "_" + str(
            node_count) + "basins_t30_y.txt"
        x_path = os.path.join(path, x_path)
        y_path = os.path.join(path, y_path)
        # 注意, time_step和generate_data中的对应起来
        time_step = 30
        x = np.loadtxt(x_path, delimiter=' ').reshape((-1, time_step, 5))
        labels = np.loadtxt(y_path, delimiter=' ').reshape((-1, 1))
        return x, labels

    def load_data(self, unit_code, node_count):
        path = self.__path
        train_x, train_y = self.load_core(path, unit_code=unit_code, node_count=node_count)
        test_x, test_y = self.load_core_test(path, unit_code=unit_code, node_count=node_count)
        return train_x, train_y, test_x, test_y

    def extract_node_data(self, all_x, all_y, block_id, batch_size, node_count):
        """抽取单个节点的数据"""
        single_data = SingleNodeData(all_x, all_y, block_id, batch_size, node_count)
        node_x, node_y = single_data.extract_data()
        return node_x, node_y


if __name__ == '__main__':
    path = "D:\\河海大学\\研究课题\\研究课题\\实验相关\\PSGD\\Parallel-SGD\\dataset\\code_test\\fed_meta_ref\\MAML\\data"
    hydro_data = HydroDataMeta(path)
    unit_code = "01"
    train_x, train_y, test_x, test_y = hydro_data.load_data(unit_code)
    block_id = 0
    node_count = 6
    batch_size = 256*node_count

    x, y = hydro_data.extract_node_data(train_x, train_y, block_id, batch_size, node_count)
    print("=========")
