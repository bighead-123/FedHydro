import numpy as np

from itertools import combinations
from profiles.blockassignment.abstract import AbsBlockAssignment


class IIDBlockAssignment(AbsBlockAssignment):

    def __init__(self, node_count, redundancy):

        super().__init__(node_count, redundancy)
        self.__block_count = int(np.math.factorial(node_count) / (
                    np.math.factorial(redundancy) * np.math.factorial(node_count - redundancy)))

        self.__block_ids = list(range(self.__block_count))

        self.__block_2_node = list(combinations(self.nodes, redundancy))
        self.__node_2_block = [[] for _ in self.nodes]

        block_id = 0

        for nodes in self.__block_2_node:
            for node in nodes:
                self.__node_2_block[node].append(block_id)
            block_id += 1

    @property
    def block_2_node(self):
        """block_2_node 属性定义了从block到node的映射，可以是数组，也可以是 dict"""
        return self.__block_2_node

    @property
    def node_2_block(self):
        """node_2_block 属性定义了从node到block的映射，可以是数组，也可以是 dict"""
        return self.__node_2_block

    @property
    def block_count(self):
        """block_count 属性返回blocks 属性返回的 block 个数"""
        return self.__block_count

    @property
    def blocks(self):
        """blocks 属性定义了所有可用的 block 的编号，返回为数组"""
        return self.__block_ids
