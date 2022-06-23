from abc import ABCMeta, abstractmethod

from numpy import ndarray


class IBatchIter(metaclass=ABCMeta):

    @property
    @abstractmethod
    def batch_size(self):
        pass

    @abstractmethod
    def iter(self, batch_no:int, block_no:int) -> slice:
        pass

    @abstractmethod
    def adjust_ratio(self, block_size_ratio:[list, tuple, ndarray]):
        pass

    @abstractmethod
    def set_block_size(self, block_size: int):
        pass

    @abstractmethod
    def set_batch_size(self, batch_size: int):
        pass

    @abstractmethod
    def set_splitters(self, block_count: int, block_size: int):
        pass
