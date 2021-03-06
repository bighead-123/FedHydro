from numpy import ndarray

from nn.interface import IOperator
from nn.abstract import AbsFlexibleBinaryNode
from nn.operation.abstract import OperandHelper


class Add(AbsFlexibleBinaryNode, OperandHelper):

    def __init__(self, op1:IOperator, op2:IOperator):
        super().__init__(op1, op2)

    def output_shape(self) -> [list, tuple, None]:
        return self.op_left.output_shape()

    def do_forward(self, left:[float, ndarray], right:[float, ndarray], training:bool=True) -> [float, ndarray]:
        return left + right

    def do_backward(self, left:[float, ndarray], right:[float, ndarray], grad:[float, ndarray]) -> [ndarray, float]:
        return grad, grad
