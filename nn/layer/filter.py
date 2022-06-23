import numpy as np
from nn.interface import IOperator
from nn.layer.abstract import AbsLayer
from nn.activation.interface import IActivation
from nn.value.trainable import Weights


class Filter(AbsLayer):
    # 加上可训练的参数
    def __init__(self, activation: IActivation = None, inputs: IOperator = None):
        super().__init__(inputs, activation)
        self.__in_shape = None
        self.__out_shape = None
        self.__in_batch_shape = None

    # 输出的维度
    def output_shape(self) -> [list, tuple, None]:
        return [-1, self.__out_shape]

    @property
    def variables(self) -> tuple:
        return ()

    # 初始化参数
    # 这里的x的shpae[1]可以看成是in_units
    def initialize_parameters(self, x) -> None:
        pass

    def do_forward_predict(self, x):
        self.__in_batch_shape = x.shape
        out = x[:, -1, :]
        return out

    def do_forward_train(self, x):
        return self.do_forward_predict(x)

    # 修改w自己的参数，其中input_ref是在前向计算时AbsLayer中F赋值的
    def backward_adjust(self, grad) -> None:
        pass

    # 给前一层返回的计算结果, grad即delta
    def backward_propagate(self, grad):
        out = np.zeros(self.__in_batch_shape)
        out[:, -1, :] = grad
        return out

    def __str__(self):
        return "<Filter Layer, Units: {}>".format(0)

    def __repr__(self):
        return "<Filter Layer, Units: {}>".format(0)
