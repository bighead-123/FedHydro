import numpy as np

from dataset.transforms.abstract import AbsTransformer


class FedImageCls(AbsTransformer):

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "<Scale to 1.0 and make one-hot label>"

    @property
    def params(self):
        return tuple()

    def run(self, train_x, train_y, test_x, test_y) -> tuple:
        return train_x, train_y, test_x, test_y
        # return (train_x - 0.5).reshape(-1,28,28,1), np.eye(62)[train_y], (test_x - 0.5).reshape(-1,28,28,1), np.eye(62)[test_y]