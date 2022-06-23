from numpy import ndarray
from typing import Set


class BlockWeight:
    """
        Weights calculated using one block
        BlockWeight是本项目传递消息的封装结构,其中content是一个列表，通常情况下存储该包发送的node_id,以及要发送的梯度
    """

    def __init__(self, content: ndarray, block_id: int):
        self.block_id = block_id
        self.content = content
