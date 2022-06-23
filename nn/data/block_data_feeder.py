from abc import ABCMeta, abstractmethod
from typing import List
from numpy import ndarray
from nn.data.interface import IDataFeeder
from profiles.interface import IBatchIter


class IPSGDBlockMgr(metaclass=ABCMeta):

    @property
    @abstractmethod
    def current_block_id(self):
        pass

    @property
    @abstractmethod
    def batch_id(self):
        pass

    @property
    @abstractmethod
    def end(self):
        pass

    # added by cyp
    @property
    @abstractmethod
    def support_batches(self):
        pass

    @abstractmethod
    def query_iter(self):
        pass

    @property
    @abstractmethod
    def query_batch_id(self):
        pass

    @property
    @abstractmethod
    def query_batches(self):
        pass
    # added by cyp end


class PSGDBlockDataFeeder(IDataFeeder, IPSGDBlockMgr):

    def __init__(self, x: ndarray, y: ndarray, batch_iter: IBatchIter, block_ids: List[int]):
        self.__x: ndarray = x
        self.__y: ndarray = y
        self.__total_blocks: List[int] = block_ids
        self.__cur_block: int = self.__total_blocks[0]
        self.__iter: int = 0
        self.__batch_id: int = 0
        self.__end: bool = False
        assert batch_iter.batch_size < len(x), \
            "Number of input samples is too small. P-SGD requires {} at least.".format(batch_iter.batch_size)
        self.__batch_size: int = batch_iter.batch_size  # 总的batch_size
        self.__batch_iter: IBatchIter = batch_iter
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
        return self.__end

    @property
    def length(self):
        return self.__batches

    @property
    def current_block_id(self):
        return self.__cur_block

    def __iter__(self):
        self.__iter = 0
        for self.__batch_id in range(self.__batches):
            self.__end = False
            for b_id in self.__total_blocks:  # [1], 仅是本worker自己的block_id
                self.__cur_block = b_id
                self.__iter += 1
                self.__end = b_id == self.__total_blocks[-1]  # 所以在目前使用范围内，每一个block，该值都为True
                sli = self.__batch_iter.iter(self.__batch_id, b_id)
                part_x = self.__x[sli]
                part_y = self.__y[sli]
                yield part_x, part_y

    # added by cyp end
    @property
    def support_batches(self):
        return self.__batches

    def query_iter(self):
        pass

    def query_batch_id(self):
        pass

    def query_batches(self):
        pass

    # added by cyp end

    def __repr__(self):
        print(self.__str__())

    def __str__(self):
        return "<P-SGD 01 iterator, current batch: {} in block: {}, total: {}."\
            .format(self.__iter, self.__cur_block, self.__batches)


class FedMetaBlockDataFeeder(IDataFeeder, IPSGDBlockMgr):

    def __init__(self, x: ndarray, y: ndarray, query_x: ndarray, query_y: ndarray, batch_iter: IBatchIter, block_ids: List[int]):
        self.__x: ndarray = x
        self.__y: ndarray = y
        self.__total_blocks: List[int] = block_ids
        self.__cur_block: int = self.__total_blocks[0]
        self.__iter: int = 0
        self.__batch_id: int = 0
        self.__end: bool = False
        assert batch_iter.batch_size < len(x), \
            "Number of input samples is too small. P-SGD requires {} at least.".format(batch_iter.batch_size)
        self.__batch_size: int = batch_iter.batch_size  # 总的batch_size
        self.__batch_iter: IBatchIter = batch_iter
        self.__batches: int = len(x) // self.__batch_size

        # 基于查询集的DataFeeder
        self.__query_x: ndarray = query_x
        self.__query_y: ndarray = query_y
        self.__query_total_blocks: List[int] = block_ids
        self.__query_cur_block: int = self.__query_total_blocks[0]
        self.__query_iter: int = 0
        self.__query_batch_id: int = 0
        self.__query_end: bool = False
        assert batch_iter.batch_size < len(query_x), \
            "Number of input query samples is too small. P-SGD requires {} at least.".format(batch_iter.batch_size)
        self.__query_batch_size: int = batch_iter.batch_size  # 总的batch_size
        self.__query_batch_iter: IBatchIter = batch_iter  # 待确定
        self.__query_batches: int = len(query_x) // self.__query_batch_size

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
        return self.__end

    @property
    def length(self):
        return self.__batches

    @property
    def current_block_id(self):
        return self.__cur_block

    def __iter__(self):
        self.__iter = 0
        for self.__batch_id in range(self.__batches):
            self.__end = False
            for b_id in self.__total_blocks:
                self.__cur_block = b_id
                self.__iter += 1
                self.__end = b_id == self.__total_blocks[-1]  # 所以在目前使用范围内，每一个block，该值都为True
                sli = self.__batch_iter.iter(self.__batch_id, b_id)
                part_x = self.__x[sli]
                part_y = self.__y[sli]
                yield part_x, part_y  # 返回part_x, part_y，并且函数会暂停，直到下次调用或者迭代停止

    @property
    def support_batches(self):
        return self.__batches

    def query_iter(self):
        self.__query_iter = 0
        for self.__query_batch_id in range(self.__query_batches):
            self.__query_end = False
            for b_id in self.__query_total_blocks:
                self.__query_cur_block = b_id
                self.__query_iter += 1
                self.__query_end = b_id == self.__query_total_blocks[-1]  # 所以在目前使用范围内，每一个block，该值都为True
                sli = self.__query_batch_iter.iter(self.__query_batch_id, b_id)
                query_part_x = self.__query_x[sli]
                query_part_y = self.__query_y[sli]
                yield query_part_x, query_part_y

    @property
    def query_batch_id(self):
        return self.__query_batch_id

    @property
    def query_batches(self):
        return self.__query_batches

    def __repr__(self):
        print(self.__str__())

    def __str__(self):
        return "<P-SGD 01 iterator, current batch: {} in block: {}, total: {}."\
            .format(self.__iter, self.__cur_block, self.__batches)