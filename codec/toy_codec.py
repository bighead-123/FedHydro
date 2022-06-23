from typing import Union, Iterable
from codec import GlobalSettings
from codec.interfaces import Codec
from codec.essential import BlockWeight
from codec.interfaces import netEncapsulation


class ToyCodec(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)
        # 保存并记录当前批次已经收到了多少份结果
        self.__global_weights = 0
        self.__current_recv = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
        send_to = GlobalSettings.get_default().get_adversary(block_weight.block_id)
        pkg = {
            "01": block_weight.content
        }
        self.__global_weights += block_weight.content
        self.__current_recv += 1
        self.__do_grad_average()
        yield netEncapsulation(send_to,pkg)

    def receive_blocks(self, content: object) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
        self.__current_recv += 1
        self.__global_weights += content['01']
        self.__do_grad_average()

    def __do_grad_average(self):
        how_much_nodes = GlobalSettings.get_default().node_count
        if self.__current_recv == how_much_nodes:
            # 执行梯度平均
            self.set_result(self.__global_weights / how_much_nodes)
            # 重设梯度值，等待下一批次的循环
            self.__global_weights = 0
            self.__current_recv = 0
