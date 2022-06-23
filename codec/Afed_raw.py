from typing import Optional, Iterable, Tuple, Union, Dict
import numpy as np
from numpy import ndarray

from codec.essential import BlockWeight
from codec.interfaces import Codec, netEncapsulation
from utils.constants import Parameter_Server
from codec import GlobalSettings
import threading


class Afed_raw_client(Codec):
    """# 异步FEDAVG +  基于worker异步通信 server异步 + 传值 +   原始的异步更新"""
    def __init__(self, node_id):
        Codec.__init__(self, node_id)
        self.__TURN = 10
        self.__local_turn = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
        self.__local_turn += 1
        if self.__local_turn % self.__TURN == 0:
            return netEncapsulation(Parameter_Server, (self.node_id, block_weight.content))
        else:
            self.set_result(block_weight.content, lambda x, y: y)

    def receive_blocks(self, content: Tuple[int, ndarray]) -> None:
        self.set_result(content[1], lambda x, y: y)
        # pass


class Afed_raw_server(Codec):

    def __init__(self, node_id):
        Codec.__init__(self, node_id)
        self.Bak_Weights = None
        # self.DROP_OUT = 0.1

    def dispose(self):
        pass
        # self.Bak_Weights_Node.clear()

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
        # 保护性if 用于防止初始化将网络参数减半
        if self.Bak_Weights is None:
            self.Bak_Weights = content[1]
        else:
            self.Bak_Weights = (self.Bak_Weights + content[1]) / 2
        return netEncapsulation(content[0], (Parameter_Server, self.Bak_Weights))

