import numpy as np

from dataset.transforms.abstract import AbsTransformer


class True_Non_IID(AbsTransformer):
    """
        Make your dataset i.i.d. compatible.
        Transform input_ref x, y into non-i.i.d. 01 distribution.

    :param batch_size: batch_size while splitting.
    :param disorder: non-iid has disorder rate, The higher the disorder, the more likely shuffle.
    :return: (x, y) with non-i.i.d. distribution
    """

    def __init__(self, batch_size, disorder):
        super().__init__()
        self.__batch_size = batch_size
        self.__disorder = disorder

    def __repr__(self):
        return "<Make non-iid dataset, based on labels>"

    @property
    def params(self):
        return self.__batch_size,

    def run(self, x, y, test_x, test_y) -> tuple:
        # get total sample count
        n = y.shape[0]
        # get real n
        n = n - n % self.__batch_size

        # get margin for each sampling point
        margin = n // self.__batch_size
        # 一列对应一个batch
        # get batch sampling point, each batch corresponds with a column.
        indicator = np.arange(0, n).reshape([self.__batch_size, margin])
        # transpose and reshape indicator
        # reshap为1行， 每batch_size个为一个batch， batch_size * margin
        indicator = np.reshape(indicator.T, newshape=-1)
        # get sorts index
        # 对数组进行从小到大排序后，输出其对应原数组的索引下标
        idx = np.argsort(y)
        # sampling 01 index according to sampling indicator
        # 按照indicator的shape取idx的前size个元素(size=indicator.size)
        idx = idx[indicator]
        # sampling read 01
        t_x,t_y = x[idx], y[idx]

        chaos_num = int(n * self.__disorder)
        index_train = [i for i in range(chaos_num)]
        np.random.shuffle(index_train)
        t_x_0 = t_x[:chaos_num]
        t_y_0 = t_y[:chaos_num]
        t_x_0 = t_x_0[index_train]
        t_y_0 = t_y_0[index_train]
        t_x = np.concatenate([t_x_0, t_x[chaos_num:]], axis=0)
        t_y = np.concatenate([t_y_0, t_y[chaos_num:]], axis=0)

        return t_x, t_y, test_x, test_y


if __name__ == '__main__':
    import dataset.fed_time_series as fed
    import dataset.fed_mnist as mnist
    import dataset.cifar as cifar
    mk = True_Non_IID(32, 0.3)
    x, y, _, = mk.run(*mnist.MNIST().load())
    # x,y,_,_ = mk.run(*mnist.MNIST().load())
    # x, y, _, _ = mk.run(*fed.TimeSeries().load())
    print(x)
    print(y)
