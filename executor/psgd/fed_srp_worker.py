import time
import pandas as pd
import nn
from codec import GlobalSettings
from codec.shared_model import SharedModel
from executor.abstract import AbsExecutor
from executor.psgd.net_package import *
from nn.data import PSGDBlockDataFeeder
from nn.model import Model
from utils.log import Logger
import numpy as np
import math
from dataset.extract_node_data import SingleNodeData


class PSGDFedSRPWorkerExecutor(AbsExecutor):

    def __init__(self, node_id, offset):
        super().__init__(node_id, offset)
        self.__log = Logger('Fit-{}'.format(node_id), log_to_file=True)
        self.__trace_filename = [self.__log.File_Name]
        # waiting for those
        self.__model: [Model] = None
        self.__optimizer: [IPSGDOpContainer] = None
        self.__batch_iter: [IBatchIter] = None
        self.__trans: [ITransfer] = None
        self.__data: [IDataset] = None
        self.__misc: [misc_package] = None
        self.__done: bool = False

        #  added for fed_srp
        self.__local_epoch = 1
        self.__train_spilt = 0.7
        self.__pre_weights = None
        self.__current_weights = None
        self.__received_model_weights_list = []
        self.__val_x = None
        self.__val_y = None

    def requests(self) -> List[object]:
        """
            先请求独立项，最后请求Transfer
        """
        return [Req.Setting,
                Req.Model,
                Req.Optimizer,
                Req.Data_Package,
                Req.Other_Stuff,
                Req.Extra_Content]

    def satisfy(self, reply: list) -> list:
        unsatisfied = []
        # check list, 检查当前executor是否已经获得执行的全部条件
        for obj in reply:

            if isinstance(obj, net_setting):
                GlobalSettings.deprecated_default_settings = obj.setting()

            if isinstance(obj, extra_package):
                GlobalSettings.global_parameters = obj.acquire()  # Extra package 作为 transfer 的前置条件
                unsatisfied.append(Req.Transfer)

            if isinstance(obj, net_model):
                self.__model = obj.model
                self.__batch_iter = obj.batch_iter

            if isinstance(obj, IPSGDOpContainer):
                self.__optimizer = obj

            if isinstance(obj, ITransfer):
                self.__trans = obj

            if isinstance(obj, misc_package):
                self.__misc = obj

            if isinstance(obj, IDataset):
                if not obj.check():
                    unsatisfied.append(Req.Data_Content)
                else:
                    self.__data = obj

        return unsatisfied

    def ready(self) -> bool:
        return self.__check()[0]

    def __check(self) -> Tuple[bool, List[str]]:
        status = []
        s1 = isinstance(self.__optimizer, IPSGDOpContainer)
        status.append("Optimizer:{}".format("OK" if s1 else "ABSENT"))
        s2 = isinstance(self.__model, IModel)
        status.append("Model:{}".format("OK" if s2 else "ABSENT"))
        s3 = isinstance(self.__data, IDataset)
        status.append("Dataset:{}".format("OK" if s3 else "ABSENT"))
        s4 = isinstance(self.__misc, misc_package)
        status.append("Others:{}".format("OK" if s4 else "ABSENT"))
        s5 = isinstance(self.__trans, ITransfer)
        status.append("Transfer:{}".format("OK" if s5 else "ABSENT"))
        s6 = isinstance(self.__batch_iter, IBatchIter)
        status.append("Batch Iterator:{}".format("OK" if s6 else "ABSENT"))
        s7 = isinstance(GlobalSettings.deprecated_default_settings, ISetting)
        status.append("Settings:{}".format("OK" if s7 else "ABSENT"))
        s8 = isinstance(GlobalSettings.global_parameters, dict)
        status.append("Extra Parameters:{}".format("OK" if s8 else "ABSENT"))
        return s1 and s2 and s3 and s4 and s5 and s6 and s7 and s8, status

    def done(self) -> bool:
        return self.__done

    def start(self, com: ICommunication_Controller) -> None:
        state, report = self.__check()
        self.__log.log_message("Ready:{} \n\t Check List:\n\t\t--> {}".format(state, "\n\t\t--> ".join(report)))
        # get dataset
        train_x, train_y, test_x, test_y = self.__data.load()
        self.__log.log_message('Dataset is ready, type: ({})'.format(self.__data))

        # build 01 feeder
        # 从总训练集数据中抽取当前节点对应的训练数据
        block_ids = GlobalSettings.get_default().node_2_block[com.Node_Id]  # [1], 就是自己的block_id号，是集合，但只有1个元素
        node_num = GlobalSettings.get_default().node_count

        # 抽取当前节点拥有的数据要在改变batch_iter前
        train_single_node_data = SingleNodeData(train_x, train_y, block_ids[-1], self.__batch_iter)
        single_train_x, single_train_y = train_single_node_data.extract_data()
        self.__train_spilt = GlobalSettings.get_params('split_rate')
        self.__local_epoch = GlobalSettings.get_params('local_epoch')
        local_batch_size = GlobalSettings.get_params('local_batch_size')  # 数据丰富流域
        scarce_batch_size = GlobalSettings.get_params('scarce_batch_size')  # 数据稀缺流域
        scarce_years = GlobalSettings.get_params('scarce_years')

        # 重新设置batch_iter，转换为只从当前节点自己的数据中进行迭代训练
        if block_ids[-1] == (node_num-1):
            # 最后一个节点作为数据稀缺流域，改变batch_iter
            self.__batch_iter.set_block_size(scarce_batch_size)
            self.__batch_iter.set_batch_size(scarce_batch_size)  # 与block_size一样，把node_count当成1
            self.__batch_iter.set_splitters(block_count=1, block_size=scarce_batch_size)
            # 对于数据稀缺流域，还需要从抽取的中单个节点数据中抽取指定数量的有限年份的数据
            single_train_x = single_train_x[:scarce_years*365]
            single_train_y = single_train_y[:scarce_years*365]
        else:
            # 数据丰富流域
            self.__batch_iter.set_block_size(local_batch_size)
            self.__batch_iter.set_batch_size(local_batch_size)
            self.__batch_iter.set_splitters(block_count=1, block_size=local_batch_size)
        # 更换slice返回模式，这样迭代器就会只从本地数据上迭代
        self.__batch_iter.transform()

        # 分割数据集，用于fedsrp
        train_x, train_y, self.__val_x, self.__val_y = self.data_split(single_train_x, single_train_y)
        feeder = PSGDBlockDataFeeder(train_x, train_y, batch_iter=self.__batch_iter, block_ids=block_ids)

        # 在调用assemble之前设置ShareModel.turn，对应AdjustTurn里的self.__TURN
        local_epoch = self.__local_epoch
        batches = feeder.length
        # 设置SharedModel里的共享变量，与codec.fed_srp以及optimizer.fed_srp_sgd里面的对应
        SharedModel.local_epoch = local_epoch
        SharedModel.one_epoch_batches = batches
        if SharedModel.turn == 1:
            SharedModel.turn = batches * local_epoch  # batches
        # assemble optimizer, ParameterAveragingOptimizer for example, block_manager
        # 对ShareModel进行赋值
        self.__optimizer.assemble(transfer=self.__trans, block_mgr=feeder)
        # compile model
        self.__model.compile(self.__optimizer)
        # summary
        summary = self.__model.summary()
        self.__log.log_message(summary)
        trace_head = '{}-N({})'.format(self.__misc.mission_title, self.node_id)
        self.__log.log_message('Model set to ready.')

        log_head = self.__log.Title
        # start !
        GlobalSettings.deprecated_global_logger = self.__log
        self.__trans.start_transfer(com, group_offset=list(self.group)[0], printer=self.__log)
        # record 01
        time_start = time.time()
        data_send_start = com.Com.bytes_sent
        data_recv_start = com.Com.bytes_read

        evaluation_history = []
        title = []
        r = {}
        # special for fed_srp
        for i in range(self.__misc.epoch):
            # 先获取经过本地e个轮次之后的当前模型
            # model_list, 4*(7, 26, 80)len = var_num, list[i] = (node_count, var.shape), eg:(7, 26, 80)
            # 应该要转换成：7*(4, 26, 80)list--> len = node_number, list[i] = (var_num, var.shape)
            print("============================current global epoch:", i+1)
            # i=0时，因为并没有得到PS下发的模型，所以返回model_list=[]
            model_list = SharedModel.get_model_weight_list()
            # 保证下一轮全局更新前，SharedModel.model_weight_list为空
            SharedModel.model_weight_list = []
            self.__received_model_weights_list = self.model_list_transform(model_list)
            # 保存当前模型参数
            # current_weights: len = var_num, list[i] = var.shape, eg:(26, 80)
            self.__current_weights = self.__model.get_weights_value()
            # 获取PS下发的其他客户端模型参数并进行本地自适应更新
            self.update_weight()
            # 记录上一轮次的模型参数
            self.__pre_weights = self.__current_weights

            # change title
            self.__log.Title = log_head + "-Epo-{}".format(i + 1)
            # 本地训练
            history = self.__model.fit(feeder, epoch=local_epoch, printer=self.__log)

            r = self.__model.evaluate(test_x, test_y)
            title = r.keys()
            row = r.values()
            self.__log.log_message('Evaluate result: {}'.format(r))
            evaluation_history.append(row)

        # record 01
        time_end = time.time()
        data_sent_end = com.Com.bytes_sent
        data_recv_end = com.Com.bytes_read

        training_history = self.__model.fit_history()
        # save training history 01
        training_name = "TR-" + trace_head + ".csv"
        training_trace = pd.DataFrame(training_history.history, columns=training_history.title)
        training_trace.to_csv(training_name, index=False)
        # save evaluation history 01
        evaluation_name = "EV-" + trace_head + ".csv"
        evaluation_trace = pd.DataFrame(evaluation_history, columns=title)
        evaluation_trace.to_csv(evaluation_name, index=False)
        # save model
        model_name = "MODEL-" + trace_head + ".model"
        # 此处需要注意，不能删除，在本地测试时可以compile为其他类型优化器
        # 防止出现不能保存的问题
        # self.__model.compile(nn.gradient_descent.SGDOptimizer(learn_rate=1e-3))
        self.__model.compile(nn.gradient_descent.ADAMOptimizer(alpha=1e-3))
        self.__model.save(model_name)
        self.__trace_filename.append(training_name)
        self.__trace_filename.append(evaluation_name)
        self.__trace_filename.append(model_name)

        self.__log.log_message('Execution complete, time: {}.'.format(time_end - time_start))
        self.__log.log_message('Execution complete, Total bytes sent: {}.'.format(data_sent_end - data_send_start))
        self.__log.log_message('Execution complete, Total bytes read: {}.'.format(data_recv_end - data_recv_start))
        self.__log.log_message('Trace file has been saved to {}.'.format(trace_head))

        # set marker
        self.__done = True
        # dispose
        self.__model.clear()
        del train_x, train_y, test_x, test_y

        # return last evaluation result
        return r

    def model_list_transform(self, model_list):
        """  # model_list, 4*(7, 26, 80)len = var_num, list[i] = (node_count, var.shape), eg:(7, 26, 80)
            # 应该要转换成：7*(4, 26, 80)list--> len = node_number, list[i] = (var_num, var.shape)"""
        if len(model_list) == 0:
            return []
        node_num = model_list[0].shape[0]
        var_num = len(model_list)
        received_model_list = []
        for j in range(node_num):
            inner_model_list = []
            for i in range(var_num):
                inner_model_list.append(model_list[i][j, :])
            received_model_list.append(inner_model_list)
        return received_model_list

    def data_split(self, single_train_x, single_train_y):
        split_rate = self.__train_spilt

        train_len = math.floor(len(single_train_x) * split_rate)
        train_x = single_train_x[:train_len]
        train_y = single_train_y[:train_len]
        # 验证集
        val_x = single_train_x[train_len:]
        val_y = single_train_y[train_len:]

        return train_x, train_y, val_x, val_y

    def update_weight(self):
        other_weights_list = self.__received_model_weights_list
        # 第一轮次
        if len(other_weights_list) == 0:
            return
        val_x = self.__val_x
        val_y = self.__val_y
        pre_weights = self.__pre_weights
        if pre_weights:
            # 使用第t-1轮本地模型计算当前验证损失
            self.__model.set_weights_value(pre_weights)
            # 长度应为clients_num-1（可以接收到其他所有客户端模型参数的理想情况下）
            old_model_loss = self.__model.evaluate_sum_loss(val_x, val_y) / len(val_y)
            # 计算加权因子
            models_factor_list = self.cal_factor(old_model_loss)
            # 对加权因子进行归一化
            models_factor_list = self.factor_scale(models_factor_list)
            # 有可能会不存在比有当前模型性能更好的其他模型
            if len(models_factor_list) > 0:
                # 调用update_weights_value方法之前，将当前模型权重置为0
                self.__model.set_weights_zeros()
                for factor, received_weight in zip(models_factor_list, self.__received_model_weights_list):
                    self.__model.update_weights_value(factor, received_weight)

    def cal_factor(self, old_model_loss):
        """计算加权因子"""
        pre_weights = self.__pre_weights  # 上一轮次模型参数
        received_weights_list = self.__received_model_weights_list
        val_x = self.__val_x
        val_y = self.__val_y

        # 计算源模型-目标模型差值
        models_factor_list = []
        for weights in received_weights_list:
            params_dif = []
            for param_source, param_target in zip(weights, pre_weights):
                params_dif.append(np.reshape((param_source - param_target), newshape=(-1)))
            params_dif = np.concatenate(params_dif)  # 向量拼接
            self.__model.set_weights_value(weights)
            source_model_loss = self.__model.evaluate_sum_loss(val_x, val_y) / len(val_y)
            # 计算其他模型的加权因子
            # np.linalg.norm()：默认求二阶范式L2
            factor = (old_model_loss - source_model_loss) / (np.linalg.norm(params_dif))
            # factor = (old_model_loss - source_model_loss) / (np.linalg.norm(params_dif) + 1e-5)
            models_factor_list.append(factor)

        return models_factor_list

    def factor_scale(self, models_factor_list):
        """对权重进行标准化"""
        # 将model_factor_list中小于0的数设置为0
        models_factor_list = np.maximum(models_factor_list, 0)
        factor_sum = np.sum(models_factor_list)
        if factor_sum > 0:
            models_factor_list = [factor/factor_sum for factor in models_factor_list]
            return models_factor_list
        else:
            return []

    def trace_files(self) -> list:
        return self.__trace_filename
