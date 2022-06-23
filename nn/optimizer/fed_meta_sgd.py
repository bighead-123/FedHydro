from nn import IOptimizer, ITrainable
from nn.data.block_data_feeder import IPSGDBlockMgr
from nn.gradient_descent.interface import IGradientDescent
from psgd.interface import ITransfer
from typing import Dict, Union
from utils.adjust_turn import AdjustTurn


class FedMetaSGDOptimizer(IOptimizer):
    """
        P-SGD Optimizer
        Interact with transfer.
        Worker之间通过传输模型梯度来决定迭代方向。
        用于联邦元学习的模型参数优化器`
        used for federated meta learning, added by cyp
    """

    def __init__(self, gradient_descent: IGradientDescent, transfer: ITransfer, block_mgr: IPSGDBlockMgr):
        self.__transfer = transfer
        self.__block_mgr = block_mgr  # FedMetaBlockDataFeeder
        self.__optimizer = gradient_descent
        self.__batch_size = 1
        self.__adjust_turn = AdjustTurn(self.__block_mgr.support_batches)  # 注意属性不需要加括号

    def optimize(self, variable: ITrainable):
        """
            1st order gradient based optimize algorithm.
            {arg min}_{x}{F(x)}
        :param variable: variable object.
        :return: None
        """
        grad = variable.get_gradient()
        if grad is None:
            """用于初始化相同的全局模型参数用，使得每个worker节点的初始参数一致"""
            initial_parameter = variable.get_value()
            self.__transfer.put_weights(content=initial_parameter, var_id=variable.id, batch_no=-1, block_id=-1)
            # 获取参数服务器下发的全局初始化模型参数
            initial_parameter = self.__transfer.get_weights(variable.id, batch_no=-1)
            variable.set_value(initial_parameter)
            print(initial_parameter)
        else:
            """
                要能区分是在支撑集还是查询集上训练.
                支撑集：本地更新权重
                查询集：基于查询损失对应梯度，和PS交互后更新权重
            """
            if variable.get_shape() != grad.shape:
                grad = grad.sum(axis=0)
            if not self.__adjust_turn.is_end():
                # 在支撑集上训练，执行本地更新
                new_parameter = variable.get_value() - self.__optimizer.delta(grad / self.__batch_size)
                self.__transfer.put_weights(new_parameter, variable.id, batch_no=3,
                                            block_id=self.__block_mgr.current_block_id)
                if self.__block_mgr.end:
                    new_parameter = self.__transfer.get_weights(variable.id, batch_no=3)
                    variable.set_value(new_parameter)
                self.__adjust_turn.set_local_turn()  # local_turn += 1
            else:
                # 上传基于查询集计算出来的梯度(考虑了batch_size)，获取更新后的全局模型参数
                # 基于查询集计算出来的梯度，已在model.abstract里除以了查询集个数
                self.__transfer.put_weights(grad/self.__batch_size, variable.id, batch_no=5, block_id=-2)
                new_parameter = self.__transfer.get_weights(variable.id, batch_no=5)
                variable.set_value(new_parameter)
                print("model parameter after global update:", variable.get_value())
                self.__adjust_turn.clear_turn()  # local_turn = 0

    def set_batch_size(self, batch_size: int):
        self.__batch_size = batch_size
