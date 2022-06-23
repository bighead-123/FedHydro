from dataset.hydro_data1 import HydroDataSet1
import numpy as np
import os


def load_core(path, basin_id, kind='train'):
    """Load hydro 01 from `path`"""
    x_path = os.path.join(path, '%s_%s_x.txt' % (kind, str(basin_id)))
    y_path = os.path.join(path, '%s_%s_y.txt' % (kind, str(basin_id)))

    x = np.loadtxt(x_path, delimiter=' ').reshape((-1, 30, 5))
    labels = np.loadtxt(y_path, delimiter=' ').reshape((-1, 1))
    return x, labels


def load_basin_data(block_size, basin_id):
    # 单个节点的数据01030500
    path = 'D:\\河海大学\\研究课题\\水文预报\\课题实验\\Parallel-SGD\\dataset\\series_data\\01'
    path = os.path.join(path, str(basin_id))
    block_size = block_size
    single_train_x, single_train_y = load_core(path, basin_id, 'train')
    single_test_x, single_test_y = load_core(path, basin_id, 'test')
    train_len = single_train_x.shape[0]
    test_len = single_test_x.shape[0]
    train_len = train_len - train_len % block_size  # 去除多余数据
    test_len = test_len - test_len % block_size
    single_train_x, single_train_y = single_train_x[:train_len], single_train_y[:train_len]
    single_test_x, single_test_y = single_test_x[:test_len], single_test_y[:test_len]

    return single_train_x, single_train_y, single_test_x, single_test_y


def compare(block_size, fed_x, fed_y, single_x, single_y):
    margin = fed_y.shape[0] // (block_size*3)
    margin1 = single_y.shape[0] // block_size  # margin = margin1
    pos = 0
    basin_num = 3
    pos_single = 0
    pos_fed = 0
    for num in range(margin):
        # # 01030500，第1个节点数据
        # pos_single = num * block_size  # 0, 256, 256*2
        # pos_fed = num*block_size * basin_num  # 0, 256*3
        # # fed_slice_y = fed_y[pos_fed: pos_fed + block_size]
        # fed_slice_x = fed_x[pos_fed: pos_fed + block_size, :, :]
        # # single_slice_y = single_y[pos_single: pos_single + block_size]
        # single_slice_x = single_x[pos_single: pos_single + block_size, :, :]
        # print(fed_slice_x - single_slice_x)

        # # 01013500，第2个节点数据
        # pos_single = num * block_size  # 0, 256, 256*2
        # pos_fed = 256 + num * block_size * basin_num  # 0, 256*3
        # # fed_slice_y = fed_y[pos_fed: pos_fed + block_size]
        # fed_slice_x = fed_x[pos_fed: pos_fed + block_size, :, :]
        # # single_slice_y = single_y[pos_single: pos_single + block_size]
        # single_slice_x = single_x[pos_single: pos_single + block_size, :, :]
        # print(fed_slice_x - single_slice_x)

        # 01031500，第3个节点数据
        pos_single = num * block_size  # 0, 256, 256*2
        pos_fed = 256*2 + num * block_size * basin_num  # 0, 256*3
        # fed_slice_y = fed_y[pos_fed: pos_fed + block_size]
        fed_slice_x = fed_x[pos_fed: pos_fed + block_size, :, :]
        # single_slice_y = single_y[pos_single: pos_single + block_size]
        single_slice_x = single_x[pos_single: pos_single + block_size, :, :]
        print(fed_slice_x - single_slice_x)


# 加载联邦数据,['01030500', '01013500', '01031500'],batch_size = 256*3, block_size=256
hydroDataSet1 = HydroDataSet1()
# fed_train_x:(16128,30,5), fed_train_y:(16128,) , fed_test_x：(5376,30,5), fed_test_y(5376,)
fed_train_x, fed_train_y, fed_test_x, fed_test_y = hydroDataSet1.load()
fed_train_x, fed_train_y, fed_test_x, fed_test_y = fed_train_x, np.reshape(fed_train_y, (-1, 1)), \
                                                   fed_test_x, np.reshape(fed_test_y, (-1, 1))
# 加载单个流域/节点数据
block_size = 256
# single_train_x, single_train_y, single_test_x, single_test_y = load_basin_data(block_size, '01013500')
single_train_x, single_train_y, single_test_x, single_test_y = load_basin_data(block_size, '01031500')

# # 01030500的数据和联合数据中对应部分对比
# compare(block_size=block_size, fed_x=fed_train_x, fed_y=fed_train_y, single_x=single_train_x, single_y=single_train_y)

# 01013500的数据和联合数据中对应部分对比
# compare(block_size=block_size, fed_x=fed_train_x, fed_y=fed_train_y, single_x=single_train_x, single_y=single_train_y)

# 01031500的数据和联合数据中对应部分对比
compare(block_size=block_size, fed_x=fed_train_x, fed_y=fed_train_y, single_x=single_train_x, single_y=single_train_y)