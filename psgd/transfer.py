from threading import Thread
from typing import Dict, Sequence
from numpy import ndarray
from network import ICommunication_Controller
from psgd.interface import ITransfer
from psgd.sync.interface import ReadTimeOut, IParallelSGD
from utils.log import IPrinter


class NTransfer(ITransfer):
    """
        string identifier used in this class.
        identifier used for json key tag.
        wrote and solved only within this code scope.
        Call start_transfer() first before send and get items.
    """
    STR_LAYER_NO = 'NLayer_NO'

    def __init__(self, weights_ctrl: Dict[int, IParallelSGD]):
        """
            构建一个传输控制器，用于在本地ML进程和远程服务器进程之间传输数据。
            每个训练进程只有一个传输transfer实例。
            build a transfer controller for transferring 01 between local ML process and
            remote server process.
            Each training process has exactly one transfer instance.
        :param weights_ctrl: list of controller for transferring 01, identified for each layer, each type.
        :param com: Network Communication Controller
        """

        # formatted as weights_initial
        self.__type_weights_controller: Dict[int, IParallelSGD] = weights_ctrl
        self.__communication_process: [ICommunication_Controller] = None

        self.__group_offset: int = 0
        self.__log: [IPrinter] = None

    def put_weights(self, content: ndarray, var_id: int, batch_no: int, block_id: int):
        """
            Put weights instantly.
            Async I/O interface was called after 01 is ready.
        """
        # Copy tag，没有达到指定通信轮次前，将权重暂存；达到指定轮次后，返回包裹好的数据：target,[node_id, content]
        update_packs = self.__type_weights_controller[var_id].update_weights(content, batch_no, block_id)
        for update_pack in update_packs:
            sender, dic = update_pack  # sender为目标节点id，由Codec返回结果确定的，dic为发送的数据
            self.__send(sender, dic, layer_no=var_id)

    def get_weights(self, var_id: int, batch_no: int) -> ndarray:
        """
            Acquire weights.
            Blocking method will wait until result is available or time limit exceed.
        """
        try:
            content, packages = self.__type_weights_controller[var_id].require_weights(batch_no)  # packages一般为[]
            for sender, dic in packages:
                self.__send(sender, dic, layer_no=var_id)
            return content
        except ReadTimeOut as e:
            if e.retry() is None:
                self.__log.log_error('Time out while getting result, retry not available.')
                raise TimeoutError('Time out while get weights.')
            for sender, dic in e.retry():
                self.__send(sender, dic, layer_no=var_id)
                self.__log.log_error('Message retry to node {}'.format(sender))

        # second chance
        try:
            content, packages = self.__type_weights_controller[var_id].require_weights(batch_no)
            for sender, dic in packages:
                self.__send(sender, dic, layer_no=var_id)
            return content
        except ReadTimeOut:
            raise TimeoutError('Time out while get weights.')

    def start_transfer(self, com: ICommunication_Controller,  group_offset: int, printer: IPrinter):
        """
            Start transferring 01 between psgd controller and communication process.
            reference call (IParallelSGD.accept_data()) without sync check, is not thread safe call.
        :param group_offset: group offset
        :param com: Communication process controller.
        :param printer: Printer class.
        """
        # acquire com
        self.__communication_process = com
        self.__log = printer
        self.__group_offset = group_offset
        working_thread = Thread(name="Transfer({})".format(com.Node_Id), target=self.__run, daemon=True)
        working_thread.start()

    def __send(self, target: Sequence[int], dic: dict, layer_no: int):
        """
            Write tag and send, 此处为往集群中发送数据实现部分
        """
        # skip none
        if len(target) == 0:
            return
        # add vlan offset，如果有多个目标则i > 0，说明为p2p形式，要往多个目标发送同一份数据
        target = [i + self.__group_offset if i >= 0 else i for i in target]
        # write tag
        dic[NTransfer.STR_LAYER_NO] = layer_no
        self.__communication_process.send_one(target, dic)

    def __run(self):
        """
            Working thread.
            Quit if self.communication_process is not alive.
        :return: None
        """
        try:
            while not self.__communication_process.is_closed():
                _, dic = self.__communication_process.get_one()
                # blocking other format
                if not isinstance(dic, dict):
                    continue
                # quit processing if the object is not sent by the class instance like NTransfer
                try:
                    var_no = dic[NTransfer.STR_LAYER_NO]
                    update_packs = self.__type_weights_controller[var_no].accept_data(dic)
                    # self.Log.log_message('Message accepted.')
                    if update_packs is None:
                        continue

                    for update_pack in update_packs:
                        sender, dic = update_pack
                        self.__send(sender, dic, var_no)
                        # self.Log.log_message('Message back to node {}'.format(sender))
                except KeyError as e:
                    # print DEBUG message
                    import sys
                    import traceback
                    exc_type, exc_value, exc_tb = sys.exc_info()
                    exc_tb = traceback.format_exception(exc_type, exc_value, exc_tb)
                    for line in exc_tb:
                        self.__log.log_message(line)
                    # print DEBUG message

        except ConnectionAbortedError as err:
            self.__log.log_message('Transfer thread exited safely.')
        except Exception as e:
            self.__log.log_error("Transfer thread reported an error:{}".format(e))
