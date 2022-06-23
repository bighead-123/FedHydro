import numpy as np

from nn.interface import IOperator
from nn.layer.abstract import AbsLayer
from nn.activation.interface import IActivation
from nn.value.trainable import Weights


class GateUnit(AbsLayer):
    # 加上可训练的参数
    def __init__(self, units, out_units, activation: IActivation = None, inputs: IOperator = None):
        super().__init__(inputs, activation)
        self.__layer_units = units
        self.__in_shape = None
        self.__out_shape = None
        self.__wx = Weights()
        self.__wh = Weights()
        self.__b = Weights()

        # 反向传播开始时，梯度为空
        self.__wx_grad = None
        self.__wh_grad = None
        self.__b_grad = None

        self.__in_batch_grad = None
        self.__hs_grad = None

        self.__hs = []  # 上一步输出
        self.__in_batch = []
        self.__h = None

    # 输出的维度
    def output_shape(self) -> [list, tuple, None]:
        return [-1, self.__layer_units]

    @property
    def variables(self) -> tuple:
        return self.__wx, self.__wh, self.__b

    # 初试化参数
    def initialize_parameters(self, x) -> None:
        shape = self.__in_shape + self.__out_shape
        self.__wx.set_value(np.random.uniform(-1, 1, shape)*0.1)
        self.__wh.set_value(np.random.uniform(-1, 1, shape)*0.1)

    '''
    hs 隐藏状态
    '''
    def forward(self, x, hs, training):
        wx = self.__wx.get_value()
        wh = self.__wh.get_value()
        b = self.__b.get_value()
        self.__h = hs
        h = hs
        out = np.dot(x, wx) + np.dot(h, wh) + b
        if training:
            # 向前传播训练时把上一个时间步的输出和当前时间步的in_batch压栈
            self.__hs.append(h)
            self.__in_batch.append(x)

            # 确保反向传播开始时参数的梯度为空
            self.__wx_grad = None
            self.__wh_grad = None
            self.__b_grad = None
        return out

    def backward(self, grad):
        wx = self.__wx.get_value()
        wh = self.__wh.get_value()
        pre_hs = self.__hs.pop()

        # 对x求偏导, 对wx求导
        in_batch = self.__in_batch.pop()
        self.__in_batch_grad = np.dot(grad, wx.T)
        wx_grad = np.dot(in_batch.T, grad)

        # 对h求导, 对wh求导
        self.__hs_grad = np.dot(grad, wh.T)
        wh_grad = np.dot(pre_hs.T, grad)

        # 对偏置项b求导
        b_grad = grad.sum(axis=0)
        if self.__wx_grad is None:
            #  当前批次第一次
            self.__wx_grad = wx_grad
        else:
            #  累积当前批次的所有梯度
            self.__wx_grad = self.__wx_grad + wx_grad

        if self.__wh_grad is None:
            self.__wh_grad = wh_grad
        else:
            self.__wh_grad = self.__wh + wh_grad

        if self.__b_grad is None:
            self.__b_grad = b_grad
        else:
            self.__b_grad = self.__b_grad + b_grad

        self.__wx.adjust(self.__wx_grad)
        self.__wh.adjust(self.__wh_grad)
        self.__b.adjust(self.__b_grad)
        return self.__in_batch_grad, self.__hs_grad

    def do_forward_predict(self, x):
        h = self.__h
        out = self.forward(x, h, False)
        return out

    def do_forward_train(self, x):
        h = self.__h
        out = self.forward(x, h, True)
        return out

    # 修改w自己的参数，其中input_ref是在前向计算时AbsLayer中F赋值的
    def backward_adjust(self, grad) -> None:
        self.backward(grad)

    # 给前一层返回的计算结果, grad即delta
    def backward_propagate(self, grad):
        return self.__in_batch_grad

    def __str__(self):
        return "<GateUnit Layer, Units: {}>".format(self.__layer_units)

    def __repr__(self):
        return "<GateUnit Layer, Units: {}>".format(self.__layer_units)
