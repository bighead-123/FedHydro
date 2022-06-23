import numpy as np
from numpy import ndarray
from typing import Optional, Iterable, Tuple, Union, Dict
from codec.essential import BlockWeight
from codec.interfaces import Codec, netEncapsulation
from nn.gradient_descent import ADAMOptimizer
from utils.constants import Parameter_Server
from codec import GlobalSettings


class FedMetaClient(Codec):
    """
        codec for federated meta learning, added by cyp
    """
    def __init__(self, node_id):
        Codec.__init__(self, node_id)
        self.__local_turn = 0
        self.__TURN = 21  # 设置成：epoch*batches

    def dispose(self):
        pass

    # 控制从该节点流出的数据的处理方式
    def update_blocks(self, block_weight: BlockWeight) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
        """
            将指定样本块生成的权重更新到集群
            :param block_weight:本项目传递消息的封装结构,其中content是一个列表，通常情况下存储该包发送的node_id,以及要发送的梯度
            :return:

        """
        if block_weight.block_id == 3:
            # 本地更新
            # 参数1：指定发送的接收者， 参数2：要发送的自定义数据（一般情况下封装自身node_id以及要发送的参数）
            self.set_result(block_weight.content)

        elif block_weight.block_id == 5:
            # 将查询集上的梯度上传至参数服务器
            return netEncapsulation(Parameter_Server, (self.node_id, block_weight.content))

        else:
            # 将初始化参数发送给参数服务器，这里的代码只有在初始化时才会执行唯一一次
            return netEncapsulation(Parameter_Server, (self.node_id, block_weight.content))

    # 控制从流入该节点的数据的处理方式， content[0]: 消息发送方, content[1]: 具体内容
    def receive_blocks(self, content: Tuple[int, ndarray]) -> None:
        """
             set result
            该方法用于更新自身操作，如果是同步的调用的话，每一轮计算后都需要执行一次set_result,set_result的方法定义如下，
            可以看到如果里面已经有值，就直接做一个相加，如果需要其他更复杂的逻辑，可以传入一个lambda表达式。
        """
        self.__local_turn = 0
        self.set_result(content[1])


class FedMetaServer(Codec):

    def __init__(self, node_id):
        Codec.__init__(self, node_id)
        self.__is_init_parameter = False
        # 此处的表示上1轮次的权重参数，第一次分发时应该为模型的初始参数，不能为0，暂定如何修改
        self.__pre_weights: Union[np.ndarray, float] = 0
        # 用于临时存储接收到的参数
        self.__bak_weights_node: Dict[int, Union[ndarray, float]] = {}
        # Adam optimizer， used for global aggregation
        self.__adam = ADAMOptimizer(5e-4)

    def dispose(self):
        self.__bak_weights_node.clear()

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
            PA Server receive a json_dict and send back a request
            从集群接收类似json的字典。
            分解对象并检查是否有足够的中间值来重建完整权重。
            可用权重将保存在self.updated_weight_buffer
        :param content:
        :return: Generator: for iterating packages to be sent with NetEncapsulation type
                None if nothing need to be sent.
        """

        # 接收worker上传的梯度， content[0]为worker节点编号，content[1]为节点上传的对应梯度，或者模型参数(初始化时)
        self.__bak_weights_node[content[0]] = content[1]

        # 参数服务器需要收齐才进行整体的更新，分发新模型
        if len(self.__bak_weights_node) == GlobalSettings.get_default().node_count:

            if self.__is_init_parameter:
                # 进入表示已完成初始化
                # 执行一步梯度下降, 需要保证上传上来的梯度已经考虑了batch_size
                average_grad = np.mean(list(self.__bak_weights_node.values()), axis=0)
                global_weight = self.__pre_weights - self.__adam.delta(average_grad)
                self.__pre_weights = global_weight
                print("ps, global_weight", global_weight)
                self.dispose()
                """  netEncapsulation：
                    该类用来包裹要发送的数据，第一个参数用来指定发送的接收者，
                    第二个参数用来发送自定义的数据，一般情况下封装自身node_id以及要发送的参数。
                """
                return netEncapsulation(GlobalSettings.get_default().nodes, (Parameter_Server, global_weight))
            else:
                initial_parameter = self.__bak_weights_node.get(content[0], 0)  # 表示要获取的值不存在时返回默认值0
                self.__pre_weights = initial_parameter
                self.dispose()
                self.__is_init_parameter = True  # 标记为已完成初始化
                return netEncapsulation(GlobalSettings.get_default().nodes, (Parameter_Server, initial_parameter))

