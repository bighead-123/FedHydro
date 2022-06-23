from nn import IOptimizer, ITrainable
from nn.data.block_data_feeder import IPSGDBlockMgr
from nn.gradient_descent.interface import IGradientDescent
from psgd.interface import ITransfer
from codec.shared_model import SharedModel
from utils.adjust_turn import AdjustTurn


class FedSRPSGDOptimizer(IOptimizer):
    """
        P-SGD Optimizer
        Interact with transfer.
        Worker之间通过传输模型梯度来决定迭代方向。
        用于FedSRP学习的模型参数优化器`
        used for federated self-adaption learning, added by cyp
    """

    def __init__(self, gradient_descent: IGradientDescent, transfer: ITransfer, block_mgr: IPSGDBlockMgr):
        self.__transfer = transfer
        self.__block_mgr = block_mgr
        self.__optimizer = gradient_descent
        self.__batch_size = 1
        # 所以fed_srp_worker里要在初始化psgd优化器前对ShareModel进行赋值
        self.__adjust_turn = AdjustTurn(SharedModel.get_batches()*SharedModel.get_local_epoch())

    def optimize(self, variable: ITrainable):
        """
            1st order gradient based optimize algorithm.
            {arg min}_{x}{F(x)}
        :param variable: variable object.
        :return: None
        """
        grad = variable.get_gradient()
        if variable.get_shape() != grad.shape:
            grad = grad.sum(axis=0)
        # 往PS上传当前第variable.id的模型参数
        new_parameter1 = variable.get_value() - self.__optimizer.delta(grad / self.__batch_size)
        self.__transfer.put_weights(new_parameter1, variable.id, batch_no=-1,
                                    block_id=self.__block_mgr.current_block_id)
        self.__adjust_turn.set_local_turn()
        if not self.__adjust_turn.is_end():
            if self.__block_mgr.end:
                # 直接进行本地更新
                new_parameter2 = self.__transfer.get_weights(variable.id, batch_no=-1)
                variable.set_value(new_parameter2)
        else:
            # 如果接收的是ps下发的，不先直接更新本地模型参数，而是先保存
            # print("==================SharedModel.model_weight_list:", SharedModel.model_weight_list)
            new_parameter = self.__transfer.get_weights(variable.id, batch_no=-1)
            SharedModel.model_weight_list.append(new_parameter)
            variable.set_value(new_parameter1)
            self.__adjust_turn.clear_turn()  # local_turn = 0

    def set_batch_size(self, batch_size: int):
        self.__batch_size = batch_size
