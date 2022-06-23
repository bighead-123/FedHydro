from abc import ABCMeta, abstractmethod
from typing import Union, Type, Dict

from numpy import ndarray

from nn.data.interface import IDataFeeder
from nn.gradient_descent.interface import IGradientDescent
from nn.loss.abstract import ILoss
from nn.metric import IMetric
from nn.optimizer import IOpContainer
from utils.log import IPrinter
from nn.model.utils import FitResultHelper


class IModel(metaclass=ABCMeta):

    @abstractmethod
    def setup(self, loss: ILoss, *metrics: IMetric):
        """
             loss and metrics
        :param loss: ILoss
        :param metrics: IMetric
        :return: None
        """
        pass

    @abstractmethod
    def compile(self, optimizer: Union[IOpContainer, Type[IGradientDescent]]):
        """
            Compile model with given optimizer
        :param optimizer: IOptimizer
        :return: None
        """
        pass

    @abstractmethod
    def global_initialization(self):
        """
        :return: None
        """
        pass

    @abstractmethod
    def get_weights_value(self):
        """
        :return: cur_weights
        """
        pass

    @abstractmethod
    def set_weights_value(self, new_weights):
        """
        :param new_weights:
        :return: None
        """
        pass

    @abstractmethod
    def update_weights_value(self, factor, received_weights):
        """
        :param factor:
        :param received_weights:
        :return:
        """
        pass

    @abstractmethod
    def update_parameter_fune(self, weights_list, fune_patten=1):
        """
        :param weights_list:
        :param fune_patten:
        :return:
        """

    @abstractmethod
    def set_weights_zeros(self):
        """
        :return:
        """

    @abstractmethod
    def get_weights_grad(self):
        """
        :return: weight grad
        """
        pass

    @abstractmethod
    def set_outer_optimiter(self, outer_optimizer):
        """
        :param outer_optimizer:
        :return:  outer_optimizer instances
        """

    @abstractmethod
    def fit(self, x: [ndarray, IDataFeeder], label: [ndarray] = None, epoch: int = 4, batch_size: int = 64,
            printer: IPrinter = None) -> FitResultHelper:
        """
            Fit model with given samples.
        :param x: ndarray or 01 feeder. requires a IDataFeeder instance or both x and label for ndarray instance.
        :param epoch: int, Epoch of training
        :param label: ndarray, Label of samples
        :param batch_size: int, batch size
        :param printer: printer type
        :return: Fitting result, contains all history records.
        """
        pass

    @abstractmethod
    def fit_early_stopping(self, x: [ndarray, IDataFeeder], label: [ndarray] = None,
                           val_x: [ndarray, IDataFeeder] = None, val_y: [ndarray] = None,
                           exp_epoch: int = 50, req_stop_num: int = 10, epoch: int = 4, batch_size: int = 64,
                           printer: IPrinter = None) -> FitResultHelper:
        """
        :param x:
        :param label:
        :param val_x:
        :param val_y:
        :param exp_epoch:
        :param req_stop_num:
        :param epoch:
        :param batch_size:
        :param printer:
        :return:
        """
        pass

    @abstractmethod
    def is_early_stoping(self, current_cpoch, is_stop, best_val_loss, stop_singal_num, best_weights,
                         val_x: [ndarray, IDataFeeder] = None, val_y: [ndarray] = None,
                         exp_epoch: int = 30, req_stop_num: int = 10):
        """
        :param current_cpoch:
        :param is_stop:
        :param best_val_loss:
        :param stop_singal_num:
        :param best_weights:
        :param val_x:
        :param val_y:
        :param exp_epoch:
        :param req_stop_num:
        :return:  is_stop, best_val_loss, stop_singal_num, best_weights
        """
        pass

    @abstractmethod
    def evaluate_sum_loss(self, x: ndarray, label: ndarray, batch_size: int = 100):
        """
        :param x:
        :param label:
        :param batch_size:
        :return:
        """
        pass

    @abstractmethod
    def fit_meta(self, x: [ndarray, IDataFeeder], label: [ndarray] = None, epoch: int = 4, batch_size: int = 64,
                 printer: IPrinter = None):

       """
            Fit model with given samples.
       :param x: ndarray or 01 feeder. requires a IDataFeeder instance or both x and label for ndarray instance.
       :param epoch: int, Epoch of training
       :param label: ndarray, Label of samples
       :param batch_size: int, batch size
       :param printer:
       :return:

       """
       pass

    @abstractmethod
    def evaluate_early_stop(self, x: ndarray, label: ndarray):
        """
        :param x:
        :param label:
        :return:
        """
        pass

    @abstractmethod
    def query_evaluate(self, x: ndarray, label: ndarray, epoch: int = 4, batch_size: int = 100):
        """
            Evaluate this model with given metric.
        :param x: input samples
        :param label: labels
        :param epoch: epoch
        :param batch_size: batch_size
        :return: evaluation result
        """
        pass

    @abstractmethod
    def fit_val(self, x: [ndarray, IDataFeeder], label: [ndarray] = None, val_x: [ndarray, IDataFeeder] = None,
                val_label: [ndarray] = None, epoch: int = 4, batch_size: int = 64, val_batch_size: int = 64,
                printer: IPrinter = None):
        """
            Fit model with given samples and valid samples. added by cyp
        :param x: ndarray or 01 feeder. requires a IDataFeeder instance or both x and label for ndarray instance.
        :param epoch: int, Epoch of training
        :param label: ndarray, Label of samples
        :param val_x: ndarray or 01 feeder. requires a IDataFeeder instance or both x and label for ndarray instance.
        :param val_label: ndarray, Label of samples
        :param batch_size: int, batch size
        :param val_batch_size: int, batch size
        :param printer: printer type
        :return: Fitting result, contains all history records.
        :return:
        """
        pass

    @abstractmethod
    def fit_history(self) -> FitResultHelper:
        """
            Get all history records.
        :return:
        """
        pass

    @abstractmethod
    def evaluate(self, x: ndarray, label: ndarray) -> Dict[str, float]:
        """
            Evaluate this model with given metric.
        :param x: input samples
        :param label: labels
        :return: evaluation result
        """
        pass

    @abstractmethod
    def predict(self, x: ndarray) -> ndarray:
        """
            Predict give input
        :param x: input samples
        :return:
        """
        pass

    @abstractmethod
    def clear(self):
        """
            Clear and reset model parameters.
        :return:
        """
        pass

    @abstractmethod
    def summary(self) -> str:
        """
            Return the summary string for this model.
        :return: String
        """
        pass
