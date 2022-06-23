import numpy as np
from nn.interface import IOperator
from nn.layer.abstract import AbsLayer
from nn.activation.interface import IActivation
from nn.value.trainable import Weights


class Dense(AbsLayer):
    # 加上可训练的参数
    def __init__(self, units, activation: IActivation = None, inputs: IOperator = None):
        super().__init__(inputs, activation)
        self.__layer_units = units
        self.__w = Weights()
        self.__b = Weights()

    # 输出的维度
    def output_shape(self) -> [list, tuple, None]:
        return [-1, self.__layer_units]

    @property
    def variables(self) -> tuple:
        return self.__w, self.__b

    # 初始化参数
    # 这里的x的shpae[1]可以看成是in_units
    def initialize_parameters(self, x) -> None:
        high = np.sqrt(6 / (x.shape[1] + self.__layer_units))
        low = -high
        self.__w.set_value(np.random.uniform(low=low, high=high, size=[x.shape[1], self.__layer_units]))
        self.__b.set_value(np.zeros(shape=[self.__layer_units]))

    def do_forward_predict(self, x):
        return np.dot(x, self.__w.get_value()) + self.__b.get_value()

    def do_forward_train(self, x):
        return self.do_forward_predict(x)

    # 修改w自己的参数，其中input_ref是在前向计算时AbsLayer中F赋值的
    def backward_adjust(self, grad) -> None:
        g_w = np.dot(self.input_ref.T, grad)
        self.__w.adjust(g_w)
        self.__b.adjust(grad)

    # 给前一层返回的计算结果
    def backward_propagate(self, grad):
        # grad(m, 1)
        g_x = np.dot(grad, self.__w.get_value().T)
        return g_x

    def __str__(self):
        return "<Dense Layer, Units: {}>".format(self.__layer_units)

    def __repr__(self):
        return "<Dense Layer, Units: {}>".format(self.__layer_units)
