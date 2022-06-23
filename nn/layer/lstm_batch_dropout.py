import numpy as np
from nn.interface import IOperator
from nn.layer.abstract import AbsLayer
from nn.activation.interface import IActivation
from nn.value.trainable import Weights
import nn


class LSTMDropout(AbsLayer):

    # input_shape(samples, timesteps, input_dim)
    def __init__(self, n_in, units,  nb_seq, h0=None, c0=None, recurrent_dropout=0, forget_bias_num=1, return_sequence=False,
                 activation: IActivation = None, inputs: IOperator = None):
        super().__init__(inputs, activation)
        self.__n_in = n_in
        self.__layer_units = units
        self.__nb_seq = nb_seq  # 时间步长
        self.__return_sequence = return_sequence
        self.__forget_bias_num = forget_bias_num
        self.__nb_batch = None
        self.__ref_mask = None
        self.__probability = recurrent_dropout
        self.__scale = 1 / (1 - recurrent_dropout)

        # 权重参数
        self.__all_w = Weights()
        self.__all_w_val, self.__d_all_w_val = None, None
        self.__h0, self.__d_h0 = h0, None
        self.__c0, self.__d_c0 = c0, None

        self.__last_output = None
        self.__IFOGf = None
        self.__IFOG = None
        self.__Hin = None
        self.__C = None  # c~
        self.__Ct = None  # c_t

        self.__training = True

    def __zeros(self, size):
        return np.array(np.zeros(size))

    def __init(self, size):
        """
            暂定
        :param size:
        :return:
        """
        # xavier_uniform初始化方式
        n_in = size[0]
        n_out = size[1]
        high = np.sqrt(6 / (n_in + n_out))
        low = -high
        return np.random.uniform(low=low, high=high, size=size)

    def __inner_init(self, size):
        """
            暂定
        :param size:
        :return:
        """
        high = np.sqrt(6 / (size[0] + size[1]))
        low = -high
        return np.random.uniform(low=low, high=high, size=size)

    def output_shape(self) -> [list, tuple, None]:
        return [-1, self.__nb_seq, self.__layer_units]

    @property
    def variables(self) -> tuple:
        return tuple([self.__all_w])

    def initialize_parameters(self, x) -> None:
        n_in = self.__n_in  # 5
        n_out = self.__layer_units  # 20
        # n_out=20, n_in=5
        # init weights, (5+20+1, 4*20)--(26, 80)
        self.__all_w_val = self.__zeros((n_in + n_out + 1, 4 * n_out))

        # bias
        if self.__forget_bias_num != 0:
            self.__all_w_val[0, n_in: 2 * n_out] = self.__forget_bias_num  # (0, 5:40)
        # Weights matrices for input x, wx
        self.__all_w_val[1:n_in + 1, n_out * 0:n_out * 1] = self.__init((n_in, n_out))  # (1:5, 20)
        self.__all_w_val[1:n_in + 1, n_out * 1:n_out * 2] = self.__init((n_in, n_out))  # (1:5, 20:40)
        self.__all_w_val[1:n_in + 1, n_out * 2:n_out * 3] = self.__init((n_in, n_out))  # (5, 40:60)
        self.__all_w_val[1:n_in + 1, n_out * 3:n_out * 4] = self.__init((n_in, n_out))  # (5, 60:80)

        # Weights matrices for memory cell, wh
        self.__all_w_val[n_in + 1:, n_out * 0:n_out * 1] = self.__inner_init((n_out, n_out))  # (5:26, 20)
        self.__all_w_val[n_in + 1:, n_out * 1:n_out * 2] = self.__inner_init((n_out, n_out))  # (5:26, 20:40)
        self.__all_w_val[n_in + 1:, n_out * 2:n_out * 3] = self.__inner_init((n_out, n_out))  # (5:26, 40:60)
        self.__all_w_val[n_in + 1:, n_out * 3:n_out * 4] = self.__inner_init((n_out, n_out))  # (5:26, 60:80)
        self.__all_w.set_value(self.__all_w_val)

    def forward(self, input):
        """Forward propagation.

               Parameters
               ----------
               input : numpy.array
                   input should be of shape (nb_batch,nb_seq,n_in)
               c0 : numpy.array or None
                   init cell state
               h0 : numpy.array or None
                   init hidden state

               Returns
               -------
               numpy.array
                   Forward results.
               """
        # checking
        assert np.ndim(input) == 3, 'Only support batch training.'
        assert input.shape[2] == self.__n_in, 'input 01 input_dim and batch_data dismatch'
        # 获取执行到当前批次时权重的值：all_w_val，待查
        self.__all_w_val = self.__all_w.get_value()
        h0 = self.__h0
        c0 = self.__c0
        n_out = self.__layer_units
        # shape
        nb_batch, nb_seq, n_in = input.shape
        self.__nb_batch = nb_batch
        self.__nb_seq = nb_seq  # 暂定, 多余

        # 01,(256, 30, 5) -----> (30, 256, 5)将第0维和第1维换位置
        input = np.transpose(input, (1, 0, 2))
        self.__c0 = self.__zeros((nb_batch, n_out)) if c0 is None else c0  # (256,20)
        self.__h0 = self.__zeros((nb_batch, n_out)) if h0 is None else h0  # (64, 20)

        # Perform the LSTM forward pass with X as the input #
        # x plus h plus bias, lol
        # xphpb=26
        xphpb = self.__all_w_val.shape[0]
        # input [1, xt, ht-1] to each tick of the LSTM,
        Hin = self.__zeros((nb_seq, nb_batch, xphpb))  # (30, 256, 26)， 输入x, preh
        # hidden representation of the LSTM (gated cell content),
        Hout = self.__zeros((nb_seq, nb_batch, n_out))  # (30, 256, 26)
        # input, forget, output, gate (IFOG), 输入门，遗忘门，输出门，细胞即时状态（20, 256, 4*20）
        IFOG = self.__zeros((nb_seq, nb_batch, n_out * 4))
        # after nonlinearity, 经过激活函数之后的输出
        IFOGf = self.__zeros((nb_seq, nb_batch, n_out * 4))
        # cell content, c~
        C = self.__zeros((nb_seq, nb_batch, n_out))
        # tanh of cell content, tanh(c~)
        Ct = self.__zeros((nb_seq, nb_batch, n_out))
        temp_h = Hout[0]
        self.__ref_mask = np.random.uniform(0, 1, size=temp_h.shape) > self.__probability
        for t in range(nb_seq):
            # concat [x,h] as input to the LSTM
            prevh = Hout[t - 1] if t > 0 else self.__h0  # h0（64, 20）
            # bias
            Hin[t, :, 0] = 1
            Hin[t, :, 1:n_in + 1] = input[t]  # 输入x， 1：n_in+1为输入x
            if self.__training:
                prevh = np.multiply(prevh, self.__ref_mask)*self.__scale  # for mc-dropout
            Hin[t, :, n_in + 1:] = prevh  # prevh, n_in+1为h
            # compute all gate activations. dots: (most work is this line)
            IFOG[t] = Hin[t].dot(self.__all_w_val)  # wx + b
            # non-linearities， 四个门的计算
            # sigmoids; these are the gates, sigmoid(wx + b)
            IFOGf[t, :, :3 * n_out] = 1.0 / (1.0 + np.exp(-IFOG[t, :, :3 * n_out]))
            # tanh
            IFOGf[t, :, 3 * n_out:] = np.tanh(IFOG[t, :, 3 * n_out:])
            # compute the cell activation
            prevc = C[t - 1] if t > 0 else self.__c0
            # c_t = f*c_t-1 + i*c~
            C[t] = IFOGf[t, :, :n_out] * IFOGf[t, :, 3 * n_out:] + \
                   IFOGf[t, :, n_out:2 * n_out] * prevc
            Ct[t] = np.tanh(C[t])
            # h_t = o_t * tanh(c_t)
            Hout[t] = IFOGf[t, :, 2 * n_out:3 * n_out] * Ct[t]

        # record
        self.__last_output = np.transpose(Hout, (1, 0, 2))
        self.__IFOGf = IFOGf
        self.__IFOG = IFOG
        self.__Hin = Hin
        self.__C = C  # c~
        self.__Ct = Ct  # c_t

        if self.__return_sequence:
            return self.__last_output  # (256, 30, 20)

        else:
            return self.__last_output[:, -1, :]  # 不返回sequence，就返回最后一个时刻

    def backward(self, grad,  dcn=None, dhn=None):

        """Backward propagation.
            参照： def backward(self, grad, dcn=None, dhn=None):
            暂定：dch, dhn

                Parameters
                ----------
                grad : numpy.array
                    Gradients propagated to this layer.
                dcn : numpy.array
                    Gradients of cell state at `n` time step.
                dhn : numpy.array
                    Gradients of hidden state at `n` time step.

                Returns
                -------
                numpy.array
                    The gradients propagated to previous layer.
                """
        # pre_grad : (256, 20), 20为LSTM层的隐藏层单元数量
        # 获取执行到当前批次时权重的值：all_w_val
        self.__all_w_val = self.__all_w.get_value()
        Hout = np.transpose(self.__last_output, (1, 0, 2))
        nb_seq, batch_size, n_out = Hout.shape  # (30, 256, 20)
        input_size = self.__all_w_val.shape[0] - n_out - 1  # -1 due to bias

        self.__d_all_w_val = self.__zeros(self.__all_w_val.shape)  # (26, 80)
        self.__d_h0 = self.__zeros((batch_size, n_out))  # (256, 20)

        # backprop the LSTM
        dIFOG = self.__zeros(self.__IFOG.shape)  # (30, 256, 80)
        dIFOGf = self.__zeros(self.__IFOGf.shape)  # (30, 256, 80)
        dHin = self.__zeros(self.__Hin.shape)  # (30, 256, 26)
        dC = self.__zeros(self.__C.shape)  # (30, 256, 20)
        layer_grad = self.__zeros((nb_seq, batch_size, input_size))  # (30, 256, 5)
        # make a copy so we don't have any funny side effects

        # prepare layer gradients
        if self.__return_sequence:
            timesteps = list(range(nb_seq))[::-1]
            assert np.ndim(grad) == 3
        else:
            timesteps = [nb_seq - 1]
            assert np.ndim(grad) == 2
            tmp = self.__zeros((batch_size, nb_seq, n_out))
            tmp[:, -1, :] = grad  # 将pre_grad赋值给最后一个时刻
            grad = tmp  #
        dHout = np.transpose(grad, (1, 0, 2)).copy()

        # carry over gradients from later
        if dcn is not None:
            dC[nb_seq - 1] += dcn.copy()
        if dhn is not None:
            dHout[nb_seq - 1] += dhn.copy()

        for t in timesteps:

            tanhCt = self.__Ct[t]
            dIFOGf[t, :, 2 * n_out:3 * n_out] = tanhCt * dHout[t]
            # backprop tanh non-linearity first then continue backprop
            dC[t] += (1 - tanhCt ** 2) * (self.__IFOGf[t, :, 2 * n_out:3 * n_out] * dHout[t])

            if t > 0:
                dIFOGf[t, :, n_out:2 * n_out] = self.__C[t - 1] * dC[t]
                dC[t - 1] += self.__IFOGf[t, :, n_out:2 * n_out] * dC[t]
            else:
                # self__.c0暂定
                dIFOGf[t, :, n_out:2 * n_out] = self.__c0 * dC[t]
                self.__d_c0 = self.__IFOGf[t, :, n_out:2 * n_out] * dC[t]
            dIFOGf[t, :, :n_out] = self.__IFOGf[t, :, 3 * n_out:] * dC[t]
            dIFOGf[t, :, 3 * n_out:] = self.__IFOGf[t, :, :n_out] * dC[t]

            # backprop activation functions，
            dIFOG[t, :, 3 * n_out:] = (1 - self.__IFOGf[t, :, 3 * n_out:] ** 2) * dIFOGf[t, :, 3 * n_out:]
            y = self.__IFOGf[t, :, :3 * n_out]
            # dsigmoid
            dIFOG[t, :, :3 * n_out] = (y * (1.0 - y)) * dIFOGf[t, :, :3 * n_out]

            # backprop matrix multiply
            self.__d_all_w_val += np.dot(self.__Hin[t].transpose(), dIFOG[t])
            dHin[t] = dIFOG[t].dot(self.__all_w_val.transpose())

            # backprop the identity transforms into Hin
            layer_grad[t] = dHin[t, :, 1:input_size + 1]
            if t > 0:
                if self.__training:
                    temp_preh = dHin[t, :, input_size + 1:]
                    temp_preh = np.multiply(temp_preh, self.__ref_mask)*self.__scale
                    dHout[t - 1, :] += temp_preh
                else:
                    dHout[t - 1, :] += dHin[t, :, input_size + 1:]
            else:
                # for mc-dropout
                if self.__training:
                    temp_h0 = dHin[t, :, input_size + 1:]
                    temp_h0 = np.multiply(temp_h0, self.__ref_mask) * self.__scale
                    self.__d_h0 += temp_h0
                else:
                    self.__d_h0 += dHin[t, :, input_size + 1:]

        # 转置后：（256, 30, 5），只有最后一个时间步有实际值
        layer_grad = np.transpose(layer_grad, (1, 0, 2))
        return layer_grad

    def do_forward_predict(self, x):
        self.__training = False
        out = self.forward(x)
        self.__training = True
        return out

    def do_forward_train(self, x):
        return self.forward(x)

    def backward_adjust(self, grad) -> None:
        self.__all_w.adjust(self.__d_all_w_val)

    def backward_propagate(self, grad):
        # 该框架在反向传播时，会先调用这个函数，然后再调用backward_adjust
        grad = self.backward(grad)
        return grad

    def __str__(self):
        return "<LSTM Layer, Units: {}>".format(self.__layer_units)

    def __repr__(self):
        print(self.__str__())


if __name__ == '__main__':
   print()
