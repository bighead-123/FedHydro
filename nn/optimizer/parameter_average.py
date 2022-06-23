from nn import IOptimizer, ITrainable
from nn.data.block_data_feeder import IPSGDBlockMgr
from nn.gradient_descent.interface import IGradientDescent
from psgd.interface import ITransfer


class ParameterAveragingOptimizer(IOptimizer):
    """
        P-SGD Optimizer
        Interact with transfer.
        Worker之间通过传输模型参数来决定迭代方向。

        1st order gradient based optimize algorithm.
        {arg min}_{x}{F(x)}
        :param variable: variable object.
        :return: None

    """

    def __init__(self, gradient_descent: IGradientDescent, transfer: ITransfer, block_mgr: IPSGDBlockMgr):
        self.__transfer = transfer
        self.__block_mgr = block_mgr  # feeder
        self.__optimizer = gradient_descent
        self.__batch_size = 1

    def optimize(self, variable: ITrainable):
        grad = variable.get_gradient()
        if variable.get_shape() != grad.shape:
            grad = grad.sum(axis=0)
        new_parameter = variable.get_value() - self.__optimizer.delta(grad / self.__batch_size)
        self.__transfer.put_weights(new_parameter, variable.id, self.__block_mgr.batch_id,
                                    self.__block_mgr.current_block_id)
        if self.__block_mgr.end:
            new_parameter = self.__transfer.get_weights(variable.id, batch_no=self.__block_mgr.batch_id)
            variable.set_value(new_parameter)

    def set_batch_size(self, batch_size: int):
        self.__batch_size = batch_size