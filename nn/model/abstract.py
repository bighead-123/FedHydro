import sys
from abc import abstractmethod
from typing import Tuple, List, Iterable, Union, Type, Dict

import numpy as np
import pickle
from numpy import ndarray
import copy
from nn.data.interface import IDataFeeder
from nn.data.numpy_data_feeder import NumpyDataFeeder
from nn.data.block_data_feeder import FedMetaBlockDataFeeder, IPSGDBlockMgr
from nn.gradient_descent.interface import IGradientDescent
from nn.interface import IOperator, IOptimizer, ITrainable, ModelState
from nn.loss.abstract import ILoss
from nn.metric import IMetric
from nn.model.interface import IModel
from nn.model.utils import FitResultHelper
from nn.optimizer import IOpContainer, OpContainer, GDOptimizer
from nn.value.placeholder import Placeholder
from utils.log import IPrinter
from nn.utils.nse_util import calc_nse
from nn.utils.random_util import get_rng


class Model(IModel):

    def __init__(self, input_shape: [Tuple[int]] = None):
        self.__placeholder_input = Placeholder(input_shape)
        self.__ref_output: [IOperator] = None
        self.__metrics: List[IMetric] = []
        self.__loss: [ILoss] = None
        self.__optimizer: [IOptimizer] = None
        self.__fit_history: FitResultHelper = FitResultHelper()

    @abstractmethod
    def trainable_variables(self) -> Iterable[ITrainable]:
        pass

    @property
    def is_setup(self):
        return isinstance(self.__loss, ILoss) and isinstance(self.__ref_output, IOperator)

    @property
    def can_fit(self):
        return self.is_setup and isinstance(self.__optimizer, IOpContainer)

    @abstractmethod
    def call(self, x: Placeholder) -> IOperator:
        pass

    @property
    def loss(self):
        return self.__loss

    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def metrics(self):
        return self.__metrics

    def setup(self, loss: ILoss, *metrics: IMetric):
        self.__ref_output = self.call(self.__placeholder_input)
        # validate model
        if self.__placeholder_input.get_shape() is not None:
            self.__placeholder_input.set_value()
            # reset and validate
            self.__ref_output.F()
        # setup loss
        self.__loss: ILoss = loss
        # setup metric
        self.__metrics = [self.__loss]
        self.__metrics.extend(metrics)
        # validate metrics and set title
        title = ["Epochs", "Batches", "in", "Total"]
        for metric in self.__metrics:
            assert isinstance(metric, IMetric), "Something cannot be interpreted as metric were passed in."
            title.append(metric.description())

        # set title
        self.__fit_history.set_fit_title(title)

    def compile(self, optimizer: Union[IOpContainer, Type[IGradientDescent], IGradientDescent]):
        # set optimizer
        if isinstance(optimizer, IOpContainer):
            self.__optimizer = optimizer
        else:
            self.__optimizer = OpContainer(GDOptimizer, optimizer)
        self.__optimizer.optimize(*self.trainable_variables())

    def __evaluate_metrics(self, y, label) -> list:
        return [metric.metric(y, label) for metric in self.__metrics]

    def global_initialization(self):
        for var in self.trainable_variables():
            var.adjust(None)

    def get_weights_value(self):
        cur_weights = []
        for var in self.trainable_variables():
            value = var.get_value()
            cur_weights.append(value)
        return cur_weights

    def set_weights_value(self, new_weights):
        i = 0
        for var in self.trainable_variables():
            var.set_value(new_weights[i])
            i += 1

    def update_weights_value(self, factor, received_weights):
        """
            调用该方法之前，应该先把所有可训练参数置为0
        """
        i = 0
        for var in self.trainable_variables():
            value = var.get_value()
            value += factor*received_weights[i]
            var.set_value(value)
            i += 1

    def set_weights_zeros(self):
        """
            把所有可训练参数置为0
        """
        for var in self.trainable_variables():
            var_shape = var.get_shape()
            zeros = np.zeros(shape=var_shape)
            var.set_value(zeros)

    def get_weights_grad(self):
        grad_set = []
        for var in self.trainable_variables():
            gradient = var.get_gradient()
            if gradient.shape != var.get_shape():
                gradient = var.get_gradient().sum(axis=0)
            grad_set.append(gradient)
        return grad_set

    def update_parameter_fune(self, weights_list, fune_patten=1):
        """用于迁移学习，更新LSTM内部门的参数"""
        if fune_patten == 1:
            """只更新输出层w, b"""
            i = 0
            weight_len = len(weights_list) - 2
            # weight = weights_list[-1]
            for var in self.trainable_variables():
                if i >= 2:
                    var.set_value(weights_list[i])
                i += 1

        elif fune_patten == 2:
            """保留LSTM单元内部部分权重，其余更新"""
            i = 0
            lstm1 = weights_list[0][0:5 + 1, :]  # 只对输入维度为5有效
            lstm2 = weights_list[1][0:20 + 1, :]  # 只对输入维度为20有效
            lstm_list = [lstm1, lstm2]
            for var in self.trainable_variables():
                if i == 0:
                    value = var.get_value()
                    value[0:5+1, :] = lstm_list[i]
                    var.set_value(value)
                if i == 1:
                    value = var.get_value()
                    value[0:20+1, :] = lstm_list[i]
                    var.set_value(value)
                i += 1
                if i >= 2:
                    return
        else:
            pass

    # 只用于元学习的方法
    def set_outer_optimiter(self, outer_optimizer):
        weights_outer_optimizers = []
        for var in self.trainable_variables():
            gd = copy.deepcopy(outer_optimizer)
            weights_outer_optimizers.append(gd)
        return weights_outer_optimizers

    def fit(self, x: [ndarray, IDataFeeder], label: [ndarray] = None, epoch: int = 4, batch_size: int = 64,
            printer: IPrinter = None) -> FitResultHelper:
        assert self.can_fit, "Model is not prepared for training."
        assert isinstance(x, IDataFeeder) or label is not None, "Fitting process requires both x and label."
        # added by cyp
        # x1 = x
        if isinstance(x, ndarray):
            x = NumpyDataFeeder(x, label, batch_size=batch_size)

        self.__optimizer.set_batch_size(x.batch_size)
        title = [metric.description() for metric in self.__metrics]

        for j in range(epoch):
            epoch_rec = np.zeros(shape=[len(title)])
            # 打乱训练数据,
            # ====the shuffled code is added by cai yan ping====
            # seed = get_rng().randint(111, 1111111)
            # iter_x = x.iter_x()
            # iter_y = x.iter_y()
            # np.random.seed(seed)
            # np.random.shuffle(iter_x)
            # np.random.seed(seed)
            # np.random.shuffle(iter_y)
            # x.set_iter_x(iter_x)
            # x.set_iter_y(iter_y)
            # =========end========================================
            for part_x, part_y in x:
                # place_holder可以看成是形参，在执行时再赋具体的值
                self.__placeholder_input.set_value(part_x)
                # do forward propagation
                y = self.__ref_output.F()
                # get loss
                grad_y, _ = self.__loss.gradient(y, part_y)
                # do backward propagation from loss
                self.__ref_output.G(grad_y)
                # record fitting process
                batch_rec = self.__evaluate_metrics(y, part_y)
                fit_rec = [j + 1, x.position, x.length, self.__fit_history.count + 1]
                fit_rec.extend(batch_rec)
                epoch_rec += np.asarray(batch_rec) / x.length

                str_formatted = self.__fit_history.append_row(fit_rec)
                if printer:
                    printer.log_message(str_formatted)
                else:
                    # get stdout
                    sys.stdout.write('\r' + str_formatted)
                    sys.stdout.flush()
            print('')
            str_formatted = ["\t{}:{:.2f}".format(name, val) for name, val in zip(title, epoch_rec)]
            print("Epoch Summary:{}".format(','.join(str_formatted)))

        return self.__fit_history

    def fit_early_stopping(self, x: [ndarray, IDataFeeder], label: [ndarray] = None, val_x: [ndarray, IDataFeeder] =None, val_y: [ndarray] = None,
                           exp_epoch: int = 50, req_stop_num: int =10, epoch: int = 4, batch_size: int = 64,
            printer: IPrinter = None) -> FitResultHelper:
        assert self.can_fit, "Model is not prepared for training."
        assert isinstance(x, IDataFeeder) or label is not None, "Fitting process requires both x and label."
        # added by cyp
        # x1 = x
        if isinstance(x, ndarray):
            x = NumpyDataFeeder(x, label, batch_size=batch_size)

        self.__optimizer.set_batch_size(x.batch_size)
        title = [metric.description() for metric in self.__metrics]
        eval_loss = 0
        stop_singal_num = 0
        best_val_loss = 0
        best_weights = []  # 用于保存可能最优的权重
        for j in range(epoch):
            epoch_rec = np.zeros(shape=[len(title)])
            # 打乱训练数据,
            # ====the shuffled code is added by cai yan ping====
            # seed = get_rng().randint(111, 1111111)
            # iter_x = x.iter_x()
            # iter_y = x.iter_y()
            # np.random.seed(seed)
            # np.random.shuffle(iter_x)
            # np.random.seed(seed)
            # np.random.shuffle(iter_y)
            # x.set_iter_x(iter_x)
            # x.set_iter_y(iter_y)
            # =========end========================================
            for part_x, part_y in x:
                # place_holder可以看成是形参，在执行时再赋具体的值
                self.__placeholder_input.set_value(part_x)
                # do forward propagation
                y = self.__ref_output.F()
                # get loss
                grad_y, _ = self.__loss.gradient(y, part_y)
                # do backward propagation from loss
                self.__ref_output.G(grad_y)
                # record fitting process
                batch_rec = self.__evaluate_metrics(y, part_y)
                fit_rec = [j + 1, x.position, x.length, self.__fit_history.count + 1]
                fit_rec.extend(batch_rec)
                epoch_rec += np.asarray(batch_rec) / x.length

                str_formatted = self.__fit_history.append_row(fit_rec)
                if printer:
                    printer.log_message(str_formatted)
                else:
                    # get stdout
                    sys.stdout.write('\r' + str_formatted)
                    sys.stdout.flush()

            print('')
            str_formatted = ["\t{}:{:.2f}".format(name, val) for name, val in zip(title, epoch_rec)]
            print("Epoch Summary:{}".format(','.join(str_formatted)))
            # early stopping, added by cyp
            if j == 0:
                eval_loss = self.evaluate_early_stop(val_x, val_y)
                best_val_loss = eval_loss
                best_weights = self.get_weights_value()
            else:
                eval_loss = self.evaluate_early_stop(val_x, val_y)
                if eval_loss < best_val_loss:
                    best_val_loss = eval_loss
                    stop_singal_num = 0
                    if j >= exp_epoch:
                        # 保存当前模型权重
                        best_weights = self.get_weights_value()
                else:
                    stop_singal_num += 1
                if stop_singal_num >= req_stop_num:
                    # 恢复val_loss最小时的权重
                    self.set_weights_value(best_weights)
                    print("=================early stopping, stop epoch:", (j+1))
                    return self.__fit_history

        return self.__fit_history

    def is_early_stoping(self, current_cpoch, is_stop, best_val_loss, stop_singal_num, best_weights, val_x: [ndarray, IDataFeeder] =None, val_y: [ndarray] = None,
                         exp_epoch: int = 30, req_stop_num: int = 10):
        j = current_cpoch
        if j == 0:
            eval_loss = self.evaluate_early_stop(val_x, val_y)
            best_val_loss = eval_loss
            best_weights = self.get_weights_value()
            is_stop = False
        else:
            eval_loss = self.evaluate_early_stop(val_x, val_y)
            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                stop_singal_num = 0
                if j >= exp_epoch:
                    # 保存当前模型权重
                    best_weights = self.get_weights_value()
            else:
                stop_singal_num += 1
            is_stop = False
            if stop_singal_num >= req_stop_num:
                # 恢复val_loss最小时的权重
                self.set_weights_value(best_weights)
                is_stop = True
                print("=================early stopping, stop epoch:", (j + 1))

        return is_stop, best_val_loss, stop_singal_num, best_weights

    def evaluate_sum_loss(self, x: ndarray, label: ndarray, batch_size: int = 100):
        """仅适用于一个metric的情况"""
        x = NumpyDataFeeder(x, label, batch_size=batch_size)
        # get stdout
        import sys
        # get title
        title = [metric.description() for metric in self.__metrics]
        # eval_recs = []
        eval_sum_loss = 0
        for part_x, part_y in x:
            # set placeholder
            self.__placeholder_input.set_value(part_x)
            # do forward propagation
            y = self.__ref_output.F(state=ModelState.Evaluating)
            # get evaluation
            eval_rec = self.__evaluate_metrics(y, part_y)
            eval_sum_loss += eval_rec[0]

        return eval_sum_loss

    def evaluate_early_stop(self, x: ndarray, label: ndarray):
        x = NumpyDataFeeder(x, label, batch_size=100)
        # get stdout
        import sys
        # get title
        title = [metric.description() for metric in self.__metrics]
        eval_recs = []
        for part_x, part_y in x:
            # set placeholder
            self.__placeholder_input.set_value(part_x)
            # do forward propagation
            y = self.__ref_output.F(state=ModelState.Evaluating)
            # get evaluation
            eval_rec = self.__evaluate_metrics(y, part_y)
            eval_recs.append(eval_rec[0])

        return np.mean(eval_recs)

    # used for fed-meta learning, added by cyp
    def fit_meta(self, x: [ndarray, IDataFeeder], label: [ndarray] = None, epoch: int = 4, batch_size: int = 64,
                 printer: IPrinter = None):
        assert self.can_fit, "Model is not prepared for training."
        assert isinstance(x, IDataFeeder) or label is not None, "Fitting process requires both x and label."
        if isinstance(x, ndarray):
            x = NumpyDataFeeder(x, label, batch_size=batch_size)
        self.__optimizer.set_batch_size(x.batch_size)
        title = [metric.description() for metric in self.__metrics]

        epoch_rec = np.zeros(shape=[len(title)])

        for part_x, part_y in x:
            # place_holder可以看成是形参，在执行时再赋具体的值
            self.__placeholder_input.set_value(part_x)
            # do forward propagation
            y = self.__ref_output.F()
            # get loss
            grad_y, _ = self.__loss.gradient(y, part_y)
            # do backward propagation from loss
            self.__ref_output.G(grad_y)
            # record fitting process
            meta_train_rec = self.__evaluate_metrics(y, part_y)
            meta_train_rec = np.asarray(meta_train_rec[0])/x.length
            if epoch % 10 == 0:
                sys.stdout.write("\riteration:{}, Support Set Evaluating: {:.2f}.".format(epoch, meta_train_rec))  # 暂定
                sys.stdout.flush()

    def query_evaluate(self, x: ndarray, label: ndarray, epoch: int = 4, batch_size: int = 10):
        assert self.is_setup, "Model hasn't setup."
        # x = NumpyDataFeeder(x, label, batch_size=batch_size)
        # get stdout
        import sys
        meta_query_x = x
        meta_query_y = label
        # set placeholder
        self.__placeholder_input.set_value(meta_query_x)
        # do forward propagation
        y = self.__ref_output.F()
        # get loss
        grad_y, _ = self.__loss.gradient(y, meta_query_y)
        self.__ref_output.G(grad_y)
        # get evaluation
        eval_rec = self.__evaluate_metrics(y, label)
        # eval_rec = np.asarray(eval_rec[0]) / len(meta_query_y)
        eval_rec = np.asarray(eval_rec[0])
        if epoch % 10 == 0:
            sys.stdout.write("\rQuery Set Evaluating: {:.2f}.".format(eval_rec))  # 暂定
            sys.stdout.flush()
            # flush a new line
            print('')
        return eval_rec

    def fit_val(self, x: [ndarray, IDataFeeder], label: [ndarray] = None, val_x: [ndarray, IDataFeeder] = None,
                val_label: [ndarray] = None, epoch: int = 4, batch_size: int = 64, val_batch_size: int = 64,
                printer: IPrinter = None):
        assert self.can_fit, "Model is not prepared for training."
        assert isinstance(x, IDataFeeder) or label is not None, "Fitting process requires both x and label."

        if isinstance(x, ndarray):
            x = NumpyDataFeeder(x, label, batch_size=batch_size)

        self.__optimizer.set_batch_size(x.batch_size)
        title = [metric.description() for metric in self.__metrics]
        val_loss_list, nse_list = [], []
        for j in range(epoch):
            epoch_rec = np.zeros(shape=[len(title)])
            for part_x, part_y in x:
                # place_holder可以看成是形参，在执行时再赋具体的值
                self.__placeholder_input.set_value(part_x)
                # do forward propagation
                y = self.__ref_output.F()
                # get loss
                grad_y, _ = self.__loss.gradient(y, part_y)
                # do backward propagation from loss
                self.__ref_output.G(grad_y)
                # record fitting process
                batch_rec = self.__evaluate_metrics(y, part_y)
                fit_rec = [j + 1, x.position, x.length, self.__fit_history.count + 1]
                fit_rec.extend(batch_rec)
                epoch_rec += np.asarray(batch_rec) / x.length

                str_formatted = self.__fit_history.append_row(fit_rec)
                if printer:
                    printer.log_message(str_formatted)
                else:
                    # get stdout
                    sys.stdout.write('\r' + str_formatted)
                    sys.stdout.flush()
            print('')
            str_formatted = ["\t{}:{:.2f}".format(name, val) for name, val in zip(title, epoch_rec)]
            print("Epoch Summary:{}".format(','.join(str_formatted)))

            # "validation process"
            if val_x is not None and val_label is not None:
                if isinstance(val_x, ndarray):
                    val_x = NumpyDataFeeder(val_x, val_label, batch_size=val_batch_size)
                valid_losses, valid_predicts, valid_targets, nse_cal = [], [], [], []
                for iter_x, iter_y in val_x:
                    # place_holder可以看成是形参，在执行时再赋具体的值
                    self.__placeholder_input.set_value(iter_x)
                    # do forward propagation
                    y_pred = self.__ref_output.F()
                    # get validation , record fitting process
                    batch_val_loss = self.__evaluate_metrics(y_pred, iter_y)
                    # got loss and predict
                    valid_losses.append(batch_val_loss)
                    # nse_cal.append(calc_nse(iter_y, y_pred))
                    valid_predicts.extend(y_pred)
                    valid_targets.extend(iter_y)
                    # output valid status
                val_epoch_loss = np.mean(valid_losses)
                val_loss_list.append(val_epoch_loss)
                # nse = np.mean(nse_cal)
                print("validation loss: %.4f" % (float(val_epoch_loss)))

        return self.__fit_history, val_loss_list

    def fit_history(self) -> FitResultHelper:
        return self.__fit_history

    def evaluate(self, x: ndarray, label: ndarray) -> Dict[str, float]:
        assert self.is_setup, "Model hasn't setup."
        x = NumpyDataFeeder(x, label, batch_size=100)
        # get stdout
        import sys
        # get title
        title = [metric.description() for metric in self.__metrics]
        eval_recs = []
        for part_x, part_y in x:
            # set placeholder
            self.__placeholder_input.set_value(part_x)
            # do forward propagation
            y = self.__ref_output.F(state=ModelState.Evaluating)
            # get evaluation
            eval_rec = self.__evaluate_metrics(y, part_y)
            eval_recs.append(eval_rec)
            str_formatted = ["\t{}:{:.2f}".format(name, val) for name, val in zip(title, np.mean(eval_recs, axis=0))]
            sys.stdout.write("\rEvaluating: {:.2f}%{}.".format(100 * x.position / x.length, ','.join(str_formatted)))
            sys.stdout.flush()
        # flush a new line
        print('')
        return dict(zip(title, np.mean(eval_recs, axis=0)))

    def predict(self, x: ndarray):
        self.__placeholder_input.set_value(x)
        y = self.call(self.__placeholder_input).F(state=ModelState.Predicting)
        return y

    def clear(self):
        for var in self.trainable_variables():
            var.reset()

    def summary(self) -> str:

        summary = '\n------------\t\tModel Summary\t\t------------\n'

        summary += "No structure description available for this model.\n"

        if self.__loss and self.__optimizer and self.__metrics:
            summary += '\t------------\t\tAppendix\t\t------------\n'
            summary += '\tLoss:\n\t\t{}\n'.format(self.__loss)
            summary += '\tOptimizer:\n\t\t{}\n'.format(self.__optimizer)
            summary += '\tMetrics:\n'
            for metric in self.__metrics:
                summary += '\t\t<Metric: {}>\n'.format(metric.description())
            summary += '\t------------\t\tAppendix\t\t------------\n'
        summary += '\n------------\t\tModel Summary\t\t------------\n'

        return summary

    def save(self, file: str):
        with open(file, 'wb') as fd:
            pickle.dump(self, fd)

    @staticmethod
    def load(file: str) -> 'Model':
        with open(file, 'rb') as fd:
            model = pickle.load(fd)
        if model.__optimizer:
            model.compile(model.__optimizer)
        return model


