import numpy as np


# 合并标签
def merge_y(data_list, block_size, margin, data_len):
    pos = 0
    result = np.zeros(shape=(data_len * len(data_list), 1))
    basin_num = len(data_list)
    for num in range(1, margin+1):
        cur_pos = pos * basin_num
        for data in data_list:
            result[cur_pos: cur_pos + block_size, :] = data[pos: pos + block_size, :]
            cur_pos = cur_pos + block_size
        pos = block_size * num  # cur_pos是相当于合并前的单个数据集而言，所以是乘block_size
    return result


if __name__ == '__main__':
    data_list = []
    data1 = np.reshape(np.asarray(list(range(1, 13))), (-1, 1))
    data2 = np.reshape(np.asarray(list(range(14, 26))), (-1, 1))
    data3 = np.reshape(np.asarray(list(range(28, 40))), (-1, 1))
    data_list.append(data1)
    data_list.append(data2)
    data_list.append(data3)
    # print(data_list)
    block_size = 3
    margin = 4
    data_len = len(data1)
    print(data_len)
    result = merge_y(data_list, block_size, margin, data_len)
    print(result)



