import numpy as np

from numpy import ndarray
from typing import Optional, Iterable, Tuple, Union, Dict
from codec.essential import BlockWeight
from codec.interfaces import Codec, netEncapsulation
from utils.constants import Parameter_Server
from codec import GlobalSettings


# 本文件力求还原最土味的最原始的FEDAVG
# 训练数据的线程每次训练之前都会修改模型参数，无论同步还是异步，同步如果没有拿到更新陷入等待，异步直接报错
# 整个类可以认为是生产者，
class FedAvgClient(Codec):

    def __init__(self, node_id):
        Codec.__init__(self, node_id)
        self.__local_turn = 0
        self.__TURN = 30
        self.__local_weight:ndarray = 0
        self.__local_new_weight:ndarray = 0

    def dispose(self):
        pass

    # 控制从该节点流出的数据的处理方式
    def update_blocks(self, block_weight: BlockWeight) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
        self.__local_turn += 1
        self.__local_new_weight -= block_weight.content
        if self.__local_turn >= self.__TURN:

            return netEncapsulation(Parameter_Server, (self.node_id, self.__local_new_weight-self.__local_weight))
        else:
            self.set_result(self.__local_new_weight)

    # 控制从该节点流入的数据的处理方式
    def receive_blocks(self, content: Tuple[int, ndarray]) -> None:
        self.__local_turn = 0
        # self.__local_new_weight = content[1]
        self.__local_new_weight += content[1]
        self.__local_weight = self.__local_new_weight
        self.set_result(self.__local_new_weight)


# class FedAvgClient(Codec):
#
#     def __init__(self, node_id):
#         Codec.__init__(self, node_id)
#         self.__local_turn = 0
#         self.__TURN = 30
#         self.__local_weight:ndarray = 0
#         self.__local_new_weight:ndarray = 0
#
#     def dispose(self):
#         pass
#
#     def update_blocks(self, block_weight: BlockWeight) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
#         self.__local_turn += 1
#         self.__local_new_weight -= block_weight.content
#         if self.__local_turn >= self.__TURN:
#
#             return netEncapsulation(Parameter_Server, (self.node_id, self.__local_new_weight-self.__local_weight))
#         else:
#             self.set_result(self.__local_new_weight)
#
#
#     def receive_blocks(self, content: Tuple[int, ndarray]) -> None:
#         self.__local_turn = 0
#        # self.__local_new_weight = content[1]
#         self.__local_new_weight = content[1]
#         self.__local_weight = content[1]
#         self.set_result(content[1])

#
# class FedAvgServer(Codec):
#
#     def __init__(self, node_id):
#         Codec.__init__(self, node_id)
#         self.Bak_Weights_Node: Dict[int, Union[ndarray, float]] = {}
#         self.local_turn = 0
#
#     def dispose(self):
#         self.Bak_Weights_Node.clear()
#
#     def update_blocks(self, block_weight: BlockWeight) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
#         """
#             PA Server Cannot update blocks!
#         :param block_weight:
#         :return:
#         """
#         pass
#
#     def receive_blocks(self, content: Tuple[int, ndarray]) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
#         """
#             PA Server receive a json_dict and send back a request
#         :param content:
#         :return:
#         """
#         # update global current state
#         # 解码
#         self.local_turn += 1
#         if content[0] not in self.Bak_Weights_Node:
#             self.Bak_Weights_Node[content[0]] = content[1]
#         else:
#             self.Bak_Weights_Node[content[0]] += content[1]
#         if self.local_turn == GlobalSettings.get_default().node_count:
#             global_weight = np.mean(list(self.Bak_Weights_Node.values()), axis=0)
#             self.local_turn = 0
#             return netEncapsulation(GlobalSettings.get_default().nodes, (Parameter_Server, global_weight))

class FedAvgServer(Codec):

    def __init__(self, node_id):
        Codec.__init__(self, node_id)
        self.Bak_Weights_Node: Dict[int, Union[ndarray, float]] = {}

    def dispose(self):
        self.Bak_Weights_Node.clear()

    def update_blocks(self, block_weight: BlockWeight) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
        """
            PA Server Cannot update blocks!
        :param block_weight:
        :return:
        """
        pass

    def receive_blocks(self, content: Tuple[int, ndarray]) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
        """
            PA Server receive a json_dict and send back a request
        :param content:
        :return:
        """
        # update global current state
        # 解码
        self.Bak_Weights_Node[content[0]] = content[1]
        if len(self.Bak_Weights_Node) == GlobalSettings.get_default().node_count:
            global_weight = np.mean(list(self.Bak_Weights_Node.values()), axis=0)
            self.dispose()
            return netEncapsulation(GlobalSettings.get_default().nodes, (Parameter_Server, global_weight))
