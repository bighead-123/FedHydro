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
class SVDPack:
    def encode(self,node_id,content):
        # print(content.shape)
        dim = content.ndim
        if dim <= 1:
            return (node_id,content)
        else:
            firstdim = content.shape[0]
            oldshape = content.shape
            U,Sigma,VT = np.linalg.svd(content.reshape(firstdim,-1))
            full = U.shape[-1]
            Uk = U[:,0:int(full/10)]
            Sk = Sigma[0:int(full/10)]
            Vk = VT[0:int(full/10),:]
            return (node_id,Uk,Sk,Vk,oldshape)
    def decode(self,pack):
        if len(pack) == 2:
            return pack[0],pack[1]
        else:
            print(pack[1].shape,np.diag(pack[2]).shape,pack[3].shape,pack[4])
            return pack[0],(pack[1]@np.diag(pack[2])@pack[3]).reshape(pack[4])


class FedAvgClientSVD(Codec):

    def __init__(self, node_id):
        Codec.__init__(self, node_id)
        self.__local_turn = 0
        self.__TURN = 30
        self.__local_weight:ndarray = 0
        self.__local_new_weight:ndarray = 0
        self.pack = SVDPack()

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
        self.__local_turn += 1
        self.__local_new_weight -= block_weight.content
        if self.__local_turn >= self.__TURN:

            return netEncapsulation(Parameter_Server, self.pack.encode(self.node_id, self.__local_new_weight-self.__local_weight))
        else:
            self.set_result(self.__local_new_weight)

    def receive_blocks(self, content: Tuple[int, ndarray]) -> None:
        self.__local_turn = 0
        # self.__local_new_weight = content[1]
        self.__local_new_weight += content[1]
        self.__local_weight = self.__local_new_weight
        self.set_result(self.__local_new_weight)


class FedAvgServerSVD(Codec):

    def __init__(self, node_id):
        Codec.__init__(self, node_id)
        self.Bak_Weights_Node: Dict[int, Union[ndarray, float]] = {}
        # self.local_turn = 0
        self.pack = SVDPack()

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
        # self.local_turn += 1
        self.record("dfsdf")
        nodeid,cont = self.pack.decode(content)

        # print(nodeid,cont.shape)
        # if nodeid not in self.Bak_Weights_Node:
        self.Bak_Weights_Node[nodeid] = cont
        # else:
        #     self.Bak_Weights_Node[nodeid] += cont
        if len(self.Bak_Weights_Node) == GlobalSettings.get_default().node_count:
            global_weight = np.mean(list(self.Bak_Weights_Node.values()), axis=0)
            # self.local_turn = 0
            self.dispose()
            return netEncapsulation(GlobalSettings.get_default().nodes, (Parameter_Server, global_weight))

