import numpy as np
from numpy import ndarray
from typing import Optional, Iterable, Tuple, Union, Dict
from codec.essential import BlockWeight
from codec.interfaces import Codec, netEncapsulation
from utils.constants import Parameter_Server
from codec import GlobalSettings
from codec.shared_model import SharedModel


# FedFomo实现
# 整个类可以认为是生产者，
class FedSRPClient(Codec):

    def __init__(self, node_id):
        Codec.__init__(self, node_id)
        self.__local_turn = 0
        # self.__TURN = 210
        self.__TURN = 9  # batch

    def dispose(self):
        pass

    # 控制从该节点流出的数据的处理方式
    """
      BlockWeight
      BlockWeight是本项目传递消息的封装结构,其中content是一个列表，通常情况下存储该包发送的node_id,以及要发送的梯度
    """
    def update_blocks(self, block_weight: BlockWeight) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
        self.__local_turn += 1

        # 超过一定轮次之后将更新发送到参数服务器
        if self.__local_turn != 0 and self.__local_turn == SharedModel.get_batches()*SharedModel.get_local_epoch():
            # 获取当前带宽
            # 根据带宽设置编码发出去
            # 包裹要发送的数据
            # 参数1：指定发送的接收者， 参数2：要发送的自定义数据（一般情况下封装自身node_id以及要发送的参数）
            # print("==========================weight:", block_weight.content)
            return netEncapsulation(Parameter_Server, (self.node_id, block_weight.content))
        else:
            self.set_result(block_weight.content)

    # 控制从流入该节点的数据的处理方式
    def receive_blocks(self, content: Tuple[int, ndarray, Dict]) -> None:
        """
            set result
            该方法用于更新自身操作，如果是同步的调用的话，每一轮计算后都需要执行一次set_result,set_result的方法定义如下，
            可以看到如果里面已经有值，就直接做一个相加，如果需要其他更复杂
            的逻辑，可以传入一个lambda表达式。
        """
        # content:(node_id, weight)
        """
        这里考虑接收一个(node_id, ndarray), 一般node_id=-2
        eg: ndarray, (7, 20, 80), 7表示节点个数, (20, 80)表示权重w的shape        
        """
        self.__local_turn = 0
        self.set_result(content[1])


class FedSRPServer(Codec):

    def __init__(self, node_id):
        Codec.__init__(self, node_id)
        # 用于临时存储接收到的参数
        self.Bak_Weights_Node: Dict[int, Union[ndarray, float]] = {}

    def dispose(self):
        self.Bak_Weights_Node.clear()

    def update_blocks(self, block_weight: BlockWeight) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
        """
         PA Server Cannot update blocks!
        将指定样本块生成的权重更新到集群，
        函数将返回一个元组，第一个元素是要发送的目标的节点ID列表，第二个元素是要发送的实际内容json。
        如果不发送任何内容，函数将返回None。更新过程完成后，它将检查是否有足够的中间值来重建完整权重。
        这种检查在同步随机梯度下降算法中无效。
        :param block_weight:
        :return:
        """
        pass

    def receive_blocks(self, content: Tuple[int, ndarray]) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
        """
            worker中不会执行该codec，只有在ps中才会执行
            PA Server receive a json_dict and send back a request
            从集群接收类似json的字典。
            分解对象并检查是否有足够的中间值来重建完整权重。
            可用权重将保存在self.updated_weight_buffer
        :param content:
        :return: Generator: for iterating packages to be sent with NetEncapsulation type
                None if nothing need to be sent.
        """
        # update global current state
        # 解码
        self.Bak_Weights_Node[content[0]] = content[1]
        # 参数服务器需要收齐才进行整体的更新，分发新模型
        if len(self.Bak_Weights_Node) == GlobalSettings.get_default().node_count:

            all_weight_list = list(self.Bak_Weights_Node.values())
            # 包含了多个节点的模型集合，为了方便，转换为ndarray形式
            # [6, weight_shape],[6, 20, 8]
            all_weight = np.asarray(all_weight_list)
            self.dispose()
            # Parameter_Server 节点编号, PS为-2
            """  netEncapsulation：
                该类用来包裹要发送的数据，第一个参数用来指定发送的接收者，
                第二个参数用来发送自定义的数据，一般情况下封装自身node_id以及要发送的参数。
            """
            return netEncapsulation(GlobalSettings.get_default().nodes, (Parameter_Server, all_weight))
