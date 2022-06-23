from abc import ABCMeta, abstractmethod
from typing import List

from numpy import ndarray

from nn.data.interface import IDataFeeder
from profiles.interface import IBatchIter
from nn.data.block_data_feeder import IPSGDBlockMgr


class SimpleDataFeeder(IDataFeeder, IPSGDBlockMgr):

    def __init__(self, x: ndarray, y: ndarray, batch_iter: IBatchIter, block_ids: List[int]):
        self.__x: ndarray = x
        self.__y: ndarray = y
        self.__iter: int = 0
        self.__batch_id: int = 0
        assert batch_iter.batch_size < len(x), \
            "Number of input samples is too small. P-SGD requires {} at least.".format(batch_iter.batch_size)
        self.__batch_size: int = batch_iter.batch_size
        self.__batches: int = len(x) // self.__batch_size

    @property
    def position(self):
        return self.__iter

    @property
    def batch_id(self):
        return self.__batch_id

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def end(self):
        return True

    @property
    def length(self):
        return self.__batches

    @property
    def current_block_id(self):
        return 0

    def __iter__(self):
        self.__iter = 0
        for self.__batch_id in range(self.__batches):
            start = self.__batch_id * self.batch_size
            sli = slice(start, start + self.batch_size)
            self.__iter += 1
            part_x = self.__x[sli]
            part_y = self.__y[sli]
            yield part_x, part_y

    def __repr__(self):
        print(self.__str__())

    def __str__(self):
        return "<Simple Parallel 01 iterator, current batch: {}, total: {}."\
            .format(self.__iter, self.__batches)
