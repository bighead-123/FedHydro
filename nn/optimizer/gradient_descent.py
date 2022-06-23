from nn.interface import IOptimizer, ITrainable
from nn.gradient_descent.interface import IGradientDescent


class GDOptimizer(IOptimizer):

    def __init__(self, gd_optimizer: IGradientDescent):
        self.__optimizer = gd_optimizer
        self.__batch_size = 1

    def optimize(self, variable: ITrainable) -> None:
        """
            1st order gradient based optimize algorithm.
            {arg min}_{x}{F(x)}
        :param variable: variable object.
        :return: None
        """
        grad = variable.get_gradient()
        # added by cyp, for avoiding messed up by fed meta learning
        if grad is None:
            return
        # 这里if判断主要是为了处理偏置项b的梯度
        if variable.get_shape() != grad.shape:
            grad = grad.sum(axis=0)
        variable.set_value(variable.get_value() - self.__optimizer.delta(grad / self.__batch_size))
        # 不加batch_size
        # variable.set_value(variable.get_value() - self.__optimizer.delta(grad))

    def set_batch_size(self, batch_size: int):
        self.__batch_size = batch_size

    def __str__(self):
        return "<GDOptimizer, Using {}>".format(self.__optimizer)
