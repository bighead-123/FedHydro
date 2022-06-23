import numpy as np
from profiles.interface import IBatchIter
from profiles.batch_iter import BatchIter
from codec import GlobalSettings
from dataset.hydro_data1 import HydroDataSet1


class SingleNodeData:
    def __init__(self, x, y, block_id, batch_iter: IBatchIter):
        self.__block_id = block_id
        self.__x = x
        self.__y = y
        self.__batch_size = batch_iter.batch_size
        self.__batches = len(x) // self.__batch_size

    def extract_data(self):
        """从所有数据中抽取当前节点对应的数据"""
        # node_count = 3
        node_count = GlobalSettings.get_default().node_count
        block_size = self.__batch_size // node_count
        d0, d1, d2 = self.__x.shape
        d0 = d0 // node_count
        return_x = np.zeros(shape=(d0, d1, d2))
        return_y = np.zeros(shape=(d0, 1))
        cur_pos = 0
        for i in range(self.__batches):
            pos = i*self.__batch_size + self.__block_id*block_size
            return_x[cur_pos: cur_pos + block_size, :, :] = self.__x[pos: pos + block_size, :, :]
            return_y[cur_pos: cur_pos + block_size, :] = self.__y[pos: pos + block_size, :]
            cur_pos += block_size

        return return_x, return_y


if __name__ == '__main__':
    hydro_data = HydroDataSet1(1)
    train_x, train_y, test_x, test_y = hydro_data.load()
    test_y = np.reshape(test_y, (-1, 1))
    batch_iter = BatchIter(256, 3)
    single_node_data = SingleNodeData(test_x, test_y, 1, batch_iter)
    result_x, result_y = single_node_data.extract_data()
    print("a")
