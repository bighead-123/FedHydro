import numpy as np

from nn.interface import IOperator
from nn.layer.abstract import AbsLayer
from nn.activation.interface import IActivation
from nn.value.trainable import Weights


class RNN(AbsLayer):
    """
        RNN 的输入形状是(m, t, in_units)
        m: batch_size
        t: 输入系列的长度
        in_units: 输入单元数, 是输入向量的维数
    """
    # 加上可训练的参数
    def __init__(self, units, stateful=False, activation: IActivation = None, inputs: IOperator = None):
        super().__init__(inputs, activation)
        self.__layer_units = units
        self.__stateful = stateful
        self.__pre_hs = None
        self.__x_grad = None

    # 输出的维度
    def output_shape(self) -> [list, tuple, None]:
        return [-1, -1, self.__layer_units]

    @property
    def variables(self) -> tuple:
        return ()

    @property
    def stateful(self):
        return self.__stateful

    # 初始化参数
    def initialize_parameters(self, x) -> None:
        pass

    def hiden_forward(self, in_batch, pre_hs, training):
        raise Exception("hiden_forward not implement!")

    def hiden_backward(self, grad):
        raise Exception("hiden_backward not implement!")

    def forward(self, x, trainnig):
        batch_size, time_step, input_dim = x.shape
        out_units = self.__layer_units
        # 所有时间步的输出
        hstatus = np.zeros((batch_size, time_step, out_units))
        # 上一步的输出
        pre_hs = self.__pre_hs
        if pre_hs is None:
            pre_hs = np.zeros((batch_size, out_units))

        # 隐藏层循环过程, 沿时间步执行
        for t in range(time_step):
            hstatus[:, t, :] = self.hiden_forward(x[:, t, :], pre_hs, trainnig)
            pre_hs = hstatus[:, t, :]

        self.__pre_hs = pre_hs
        # pdb.set_trace()
        if not self.stateful:
            self.__pre_hs = None

        return hstatus

    def backward(self, grad):
        batch_size, time_step, input_dim = grad.shape

        # 这里的in_units是多少
        in_units = self.__layer_units
        self.__x_grad = np.zeros((batch_size, time_step, in_units))
        # pdb.set_trace()
        for t in range(time_step - 1, -1, -1):
            self.__x_grad[:, t, :], hs_grad = self.hiden_backward(grad[:, t, :])
            # pdb.set_trace()
            if t - 1 >= 0:
                grad[:, t - 1, :] = grad[:, t - 1, :] + hs_grad
        return self.__x_grad

    def do_forward_predict(self, x):
        return x

    def do_forward_train(self, x):
        out = self.forward(x, True)
        return out

    # 修改w自己的参数，其中input_ref是在前向计算时AbsLayer中F赋值的
    def backward_adjust(self, grad) -> None:
        pass

    # 给前一层返回的计算结果, grad即delta
    def backward_propagate(self, grad):
        grad_res = self.backward(grad)
        return grad_res

    def __str__(self):
        return "<RNN Layer, Units: {}>".format(self.__layer_units)

    def __repr__(self):
        return "<RNN Layer, Units: {}>".format(self.__layer_units)
