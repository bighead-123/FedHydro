import numpy as np

from nn.interface import IOperator
from nn.layer.abstract import AbsLayer
from nn.activation.interface import IActivation
from nn.value.trainable import Weights


class LSTM(AbsLayer):

    # input_shape(samples, timesteps, input_dim)
    def __init__(self, units, stateful=False, activation: IActivation = None, inputs: IOperator = None):
        super().__init__(inputs, activation)
        self.__input_units = None
        self.__layer_units = units
        # 遗忘门参数
        self.__wx_f = Weights()
        self.__wh_f = Weights()
        self.__b_f = Weights()

        # 输入门参数
        self.__wx_i = Weights()
        self.__wh_i = Weights()
        self.__b_i = Weights()

        # 输出门参数
        self.__wx_o = Weights()
        self.__wh_o = Weights()
        self.__b_o = Weights()

        # 即时单元状态参数
        self.__wx_c = Weights()
        self.__wh_c = Weights()
        self.__b_c = Weights()

        # 三类参数对应的梯度, 最终为当前批次的所有时间步的累加
        self.__wx_f_grad = None
        self.__wh_f_grad = None
        self.__b_f_grad = None

        self.__wx_i_grad = None
        self.__wh_i_grad = None
        self.__b_i_grad = None

        self.__wx_o_grad = None
        self.__wh_o_grad = None
        self.__b_o_grad = None

        self.__wx_c_grad = None
        self.__wh_c_grad = None
        self.__b_c_grad = None

        self.__stateful = stateful
        self.__pre_hs = None

        # 输入门
        self.__g_in = GateUnit(activation=Sigmoid())
        # 遗忘门
        self.__g_forget = GateUnit(activation=Sigmoid())
        # 输出门
        self.__g_out = GateUnit(activation=Sigmoid())

        # 记忆门, 即时单元状态
        self.__g_memory = GateUnit(activation=Tanh())

        # 记忆单元， c_t = f*c_(t-1) + i*~c, tanh(c_t)
        self.__memory_unit = LSTMMemoryUnit(activation=Tanh())

        # 输出单元
        self.__out_unit = LSTMOutUnit()

        # 定义当前批次权重参数的值
        self.__wx_i_val = None
        self.__wh_i_val = None
        self.__b_i_val = None

        self.__wx_f_val = None
        self.__wh_f_val = None
        self.__b_f_val = None

        self.__wx_o_val = None
        self.__wh_o_val = None
        self.__b_o_val = None

        self.__wx_c_val = None
        self.__wh_c_val = None
        self.__b_c_val = None

    def output_shape(self) -> [list, tuple, None]:
        return [-1, 1, self.__layer_units]

    @property
    def variables(self) -> tuple:
        return self.__wx_f, self.__wh_f, self.__b_f, \
               self.__wx_i, self.__wh_i, self.__b_i, \
               self.__wx_o, self.__wh_o, self.__b_o, \
               self.__wx_c, self.__wh_c, self.__b_c

    def initialize_parameters(self, x) -> None:
        in_shape = (x.shape[-1], )
        self.__input_units = x.shape[-1]
        out_shape = (self.__layer_units, )
        shape1 = in_shape + out_shape  # (5, 50)
        shape2 = out_shape + out_shape  # (50, 50)
        # xavier_uniform初始化方式
        high1 = np.sqrt(6 / (x.shape[1] + self.__layer_units))
        low1 = -high1

        high2 = np.sqrt(6 / (self.__layer_units + self.__layer_units))
        low2 = -high2

        self.__wx_f.set_value(np.random.uniform(low=low1, high=high1, size=shape1))
        self.__wh_f.set_value(np.random.uniform(low=low2, high=high2, size=shape2))
        # 修改遗忘门偏置初始值
        self.__b_f.set_value(np.ones(shape2[1]))

        self.__wx_i.set_value(np.random.uniform(low=low1, high=high1, size=shape1))
        self.__wh_i.set_value(np.random.uniform(low=low2, high=high2, size=shape2))
        self.__b_i.set_value(np.zeros(shape2[1], ))

        self.__wx_o.set_value(np.random.uniform(low=low1, high=high1, size=shape1))
        self.__wh_o.set_value(np.random.uniform(low=low2, high=high2, size=shape2))
        self.__b_o.set_value(np.zeros(shape2[1], ))
        # self.__wx_o.set_value(np.random.uniform(-1, 1, shape1) * 0.01)
        # self.__wh_o.set_value(np.random.uniform(-1, 1, shape2) * 0.01)
        # self.__b_o.set_value(np.zeros(shape2[1], ))

        self.__wx_c.set_value(np.random.uniform(low=low1, high=high1, size=shape1))
        self.__wh_c.set_value(np.random.uniform(low=low2, high=high2, size=shape2))
        self.__b_c.set_value(np.zeros(shape2[1], ))

    def get_grad(self) -> tuple:
        return self.__wx_f_grad, self.__wh_f_grad, self.__b_f_grad, \
                self.__wx_i_grad, self.__wh_i_grad, self.__b_i_grad,\
                self.__wx_o_grad, self.__wh_o_grad, self.__b_o_grad,\
                self.__wx_c_grad, self.__wh_c_grad, self.__b_c_grad

    def forward(self, x, trainning):
        # 确保当前批次反向传播开始时参数梯度为空
        if trainning:
            self.__wx_f_grad = None
            self.__wh_f_grad = None
            self.__b_f_grad = None

            self.__wx_i_grad = None
            self.__wh_i_grad = None
            self.__b_i_grad = None

            self.__wx_o_grad = None
            self.__wh_o_grad = None
            self.__b_o_grad = None

            self.__wx_c_grad = None
            self.__wh_c_grad = None
            self.__b_c_grad = None

        # 如果staeful为None
        self.__memory_unit.clear_memory()
        # 例：(400, 1, 5)
        batch_size, time_step, input_dim = x.shape
        out_units = self.__layer_units

        # 所有时间步的输出
        hstatus = np.zeros((batch_size, time_step, out_units))

        # 上一步的输出
        pre_hs = self.__pre_hs
        # 第一个时间步之前，pre_hs为None，所以先进行初始化
        if pre_hs is None:
            pre_hs = np.zeros((batch_size, out_units))

        # 获取当前训练批次权重参数的值，不同时间步共享当前批次的权重参数值
        self.__wx_i_val = self.__wx_i.get_value()
        self.__wh_i_val = self.__wh_i.get_value()
        self.__b_i_val = self.__b_i.get_value()

        self.__wx_f_val = self.__wx_f.get_value()
        self.__wh_f_val = self.__wh_f.get_value()
        self.__b_f_val = self.__b_f.get_value()

        self.__wx_o_val = self.__wx_o.get_value()
        self.__wh_o_val = self.__wh_o.get_value()
        self.__b_o_val = self.__b_o.get_value()

        self.__wx_c_val = self.__wx_c.get_value()
        self.__wh_c_val = self.__wh_c.get_value()
        self.__b_c_val = self.__b_c.get_value()

        # 隐藏层循环过程, 沿时间步执行，并记录每一个时间步的输出
        # LSTM在时间步上共享参数
        for t in range(time_step):
            # out_units：50
            # x[:, t, :]：(32, 5), pre_hs：(32, 50)
            # hstatus:(32, 1, 50)
            hstatus[:, t, :] = self.hiden_forward(x[:, t, :], pre_hs, trainning)
            pre_hs = hstatus[:, t, :]

        self.__pre_hs = pre_hs
        # 当前批次训练完，判断是否将当前批次的输出保存到下一批次作为输入，一般不保留
        if not self.__stateful:
            self.__pre_hs = None
        out = hstatus
        return out

    def backward(self, grad):
        batch_size, time_step, input_dim = grad.shape

        """# 这里的in_units是多少？"""
        in_units = self.__input_units
        x_grad = np.zeros((batch_size, time_step, in_units))

        for t in range(time_step - 1, -1, -1):
            x_grad[:, t, :], hs_grad = self.hiden_backward(grad[:, t, :])
            # pdb.set_trace()
            if t - 1 >= 0:
                grad[:, t - 1, :] = grad[:, t - 1, :] + hs_grad

        # 12/14 增加梯度裁剪功能
        # for grad in self.get_grad():
        #     np.clip(grad, -1, 1)

        return x_grad

    def hiden_forward(self, x, hs, training):

        # hs是上一时刻的输出
        # 执行输入门计算逻辑
        g_in = self.__g_in.forward(x, hs, self.__wx_i_val, self.__wh_i_val, self.__b_i_val, training)

        # 执行遗忘门计算逻辑
        g_forget = self.__g_forget.forward(x, hs, self.__wx_f_val, self.__wh_f_val, self.__b_f_val, training)

        # 执行输出门计算逻辑
        g_out = self.__g_out.forward(x, hs, self.__wx_o_val, self.__wh_o_val, self.__b_o_val, training)

        # 执行即时单元计算逻辑，得到~c
        g_memory = self.__g_memory.forward(x, hs, self.__wx_c_val, self.__wh_c_val, self.__b_c_val, training)

        # 执行tanh(f * c_(t - 1) + i * ~c)，得到c_t
        memory = self.__memory_unit.forward(g_forget, g_in, g_memory, training)

        # 计算得到当前时刻的输出h_t：o*c_t
        cur_hs = self.__out_unit.forward(g_out, memory, training)
        return cur_hs

    def hiden_backward(self, grad):
        # 基于o_t * memory(=tanh(c_t)), 计算输出门和当前细胞状态c_t的梯度
        grad_out, grad_memory = self.__out_unit.backward(grad)
        # 基于tanh(f * c_(t - 1) + i * ~c)，求遗忘门、输入门、即时细胞状态对应的梯度
        grad_forget, grad_in, grad_gm = self.__memory_unit.backward(grad_memory)

        # 以下将4个门的计算出来的梯度累加，以返回给上一个时间步
        # 将每一个时间步，三个参数的对应梯度相加，以为了梯度更新
        # 基于sigmoid(wx_o*x_o + wh_o*h_o + b_o)，计算
        wx_c = self.__wx_c_val
        wh_c = self.__wh_c_val
        grad_in_batch, grad_hs, wx_c_grad, wh_c_grad, b_c_grad = self.__g_memory.backward(wx_c, wh_c, grad_gm)
        if self.__wx_c_grad is None:
            self.__wx_c_grad = wx_c_grad
        else:
            self.__wx_c_grad = self.__wx_c_grad + wx_c_grad

        if self.__wh_c_grad is None:
            self.__wh_c_grad = wh_c_grad
        else:
            self.__wh_c_grad = self.__wh_c_grad + wh_c_grad

        if self.__b_c_grad is None:
            self.__b_c_grad = b_c_grad
        else:
            self.__b_c_grad = self.__b_c_grad + b_c_grad

        wx_o = self.__wx_o_val
        wh_o = self.__wh_o_val
        tmp1, tmp2, wx_o_grad, wh_o_grad, b_o_grad = self.__g_out.backward(wx_o, wh_o, grad_out)
        grad_in_batch += tmp1
        grad_hs += tmp2
        if self.__wx_o_grad is None:
            self.__wx_o_grad = wx_o_grad
        else:
            self.__wx_o_grad = self.__wx_o_grad + wx_o_grad

        if self.__wh_o_grad is None:
            self.__wh_o_grad = wh_o_grad
        else:
            self.__wh_o_grad = self.__wh_o_grad + wh_o_grad

        if self.__b_o_grad is None:
            self.__b_o_grad = b_o_grad
        else:
            self.__b_o_grad = self.__b_o_grad + b_o_grad

        wx_f = self.__wx_f_val
        wh_f = self.__wh_f_val
        tmp1, tmp2, wx_f_grad, wh_f_grad, b_f_grad = self.__g_forget.backward(wx_f, wh_f, grad_forget)
        grad_in_batch += tmp1
        grad_hs += tmp2
        if self.__wx_f_grad is None:
            self.__wx_f_grad = wx_f_grad
        else:
            self.__wx_f_grad = self.__wx_f_grad + wx_f_grad

        if self.__wh_f_grad is None:
            self.__wh_f_grad = wh_f_grad
        else:
            self.__wh_f_grad = self.__wh_f_grad + wh_f_grad

        if self.__b_f_grad is None:
            self.__b_f_grad = b_f_grad
        else:
            self.__b_f_grad = self.__b_f_grad + b_f_grad

        wx_i = self.__wx_i_val
        wh_i = self.__wh_i_val
        tmp1, tmp2, wx_i_grad, wh_i_grad, b_i_grad = self.__g_in.backward(wx_i, wh_i, grad_in)
        grad_in_batch += tmp1
        grad_hs += tmp2
        if self.__wx_i_grad is None:
            self.__wx_i_grad = wx_i_grad
        else:
            self.__wx_i_grad = self.__wx_i_grad + wx_i_grad

        if self.__wh_i_grad is None:
            self.__wh_i_grad = wh_i_grad
        else:
            self.__wh_i_grad = self.__wh_i_grad + wh_i_grad

        if self.__b_i_grad is None:
            self.__b_i_grad = b_i_grad
        else:
            self.__b_i_grad = self.__b_i_grad + b_i_grad

        return grad_in_batch, grad_hs

    def do_forward_predict(self, x):
        out = self.forward(x, False)
        return out

    def do_forward_train(self, x):
        out = self.forward(x, True)
        return out

    def backward_adjust(self, grad) -> None:
        self.__wx_i.adjust(self.__wx_i_grad)
        self.__wh_i.adjust(self.__wh_i_grad)
        self.__b_i.adjust(self.__b_i_grad)

        self.__wx_f.adjust(self.__wx_f_grad)
        self.__wh_f.adjust(self.__wh_f_grad)
        self.__b_f.adjust(self.__b_f_grad)

        self.__wx_o.adjust(self.__wx_o_grad)
        self.__wh_o.adjust(self.__wh_o_grad)
        self.__b_o.adjust(self.__b_o_grad)

        self.__wx_c.adjust(self.__wx_c_grad)
        self.__wh_c.adjust(self.__wh_c_grad)
        self.__b_c.adjust(self.__b_c_grad)

    def backward_propagate(self, grad):
        # 该框架在反向传播时，会先调用这个函数，然后再调用backward_adjust
        x_grad = self.backward(grad)
        return x_grad

    def __str__(self):
        return "<LSTM Layer, Units: {}>".format(self.__layer_units)

    def __repr__(self):
        print(self.__str__())


class GateUnit:

    def __init__(self, activation):
        self.__activation = activation
        self.__in_batch_grad = None
        self.__hs_grad = None

        self.__hs = []  # 上一步输出
        self.__in_batch = []

    """hs 隐藏状态"""
    def forward(self, x, h, wx, wh, b, training):
        # 例：out_units：50
        # x：(32, 5), pre_hs：(32, 50)
        # wx:(5, 50), wh:(50, 50)
        # b:(50)
        # out:(32， 50)
        out = np.dot(x, wx) + np.dot(h, wh) + b
        if training:
            # 前向传播训练时把上一个时间步的输出和当前时间步的in_batch压栈
            self.__hs.append(h)
            self.__in_batch.append(x)

            # # 确保反向传播开始时参数的梯度为空
            # self.__wx_grad = None
            # self.__wh_grad = None
            # self.__b_grad = None
        return self.__activation.do_forward(out, training)

    def backward(self, wx, wh, grad):
        grad = self.__activation.do_backward(grad)
        # wx = self.__wx
        # wh = self.__wh
        pre_hs = self.__hs.pop()
        in_batch = self.__in_batch.pop()

        # 对x求偏导, 对wx求导
        in_batch_grad = np.dot(grad, wx.T)
        wx_grad = np.dot(in_batch.T, grad)

        # 对h求导, 对wh求导
        hs_grad = np.dot(grad, wh.T)
        wh_grad = np.dot(pre_hs.T, grad)

        # 对偏置项b求导
        b_grad = grad.sum(axis=0)

        return in_batch_grad, hs_grad, wx_grad, wh_grad, b_grad


class LSTMMemoryUnit:
    """LSTM即时记忆单元
       计算tanh(c_t)
    """

    def __init__(self, activation):
        self.__activation = activation
        self.__memories = []
        self.__forgets = []
        self.__inputs = []
        self.__mcs = []

        self.__pre_mem = None
        self.__pre_mem_grad = None

    def forward(self, forget, input, memory_choice, training):
        if self.__pre_mem is None:
            self.__pre_mem = np.zeros(forget.shape)

        # c_t = f*c_t-1 + i*c_t~
        cur_m = forget * self.__pre_mem + input * memory_choice

        if training:
            self.__memories.append(self.__pre_mem)
            self.__forgets.append(forget)
            self.__inputs.append(input)
            self.__mcs.append(memory_choice)
        self.__pre_mem = cur_m

        # tanh(c_t)
        return self.__activation.do_forward(cur_m, training)

    def backward(self, gradient):
        # 注：这里的第一个参数用不到，所以用1代替
        grad = self.__activation.do_backward(gradient)

        if self.__pre_mem_grad is not None:
            grad += self.__pre_mem_grad

        pre_m = self.__memories.pop()
        forget = self.__forgets.pop()
        input = self.__inputs.pop()
        mc = self.__mcs.pop()

        self.__pre_mem_grad = grad * forget
        grad_forget = grad * pre_m
        grad_input = grad * mc
        grad_mc = grad * input

        return grad_forget, grad_input, grad_mc

    def clear_memory(self):
        self.__pre_mem = None
        self.__pre_mem_grad = None

    def reset(self):
        self.__memories = []
        self.__forgets = []
        self.__inputs = []
        self.__mcs = []

        self.clear_memory()


class LSTMOutUnit:
    """LSTM 输出单元"""

    def __init__(self):
        self.__outs = []
        self.__memories = []

    def forward(self, out, memory, training):
        res = out * memory

        if training:
            self.__outs.append(out)
            self.__memories.append(memory)

        return res

    # memory是经过tanh计算后得到的
    def backward(self, gradient):
        out = self.__outs.pop()
        memory = self.__memories.pop()

        grad_out = gradient * memory
        grad_memory = gradient * out

        return grad_out, grad_memory

    def reset(self):
        self.__outs = []
        self.__memories = []


class Sigmoid:
    def __init__(self):
        self.__grad = None

    def do_forward(self, x, trainning):
        out = 1.0 / (1.0 + np.exp(-x))
        self.__grad = out * (1 - out)
        return out

    def do_backward(self, grad):
        return grad * self.__grad


class Tanh:
    def __init__(self):
        self.__grad = None

    def do_forward(self, x, trainnig):
        out = np.tanh(x)
        self.__grad = 1 - out ** 2
        return out

    def do_backward(self, grad):
        return grad * self.__grad


if __name__ == '__main__':
    print()
