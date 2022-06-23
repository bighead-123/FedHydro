import time

from executor.abstract import AbsExecutor
from executor.interface import IExecutor
from models import *
from network import ICommunication_Controller, Serve
from network.communications import get_repr
from utils.constants import Initialization_Server, Init_Job_Submission_Timeout_Limit_Sec, VERSION
from utils.log import Logger


class Worker:

    def __init__(self):
        self.client_logger = Logger(title_info='Worker-{}'.format(get_repr()), log_to_file=True)
        self.client_logger.log_message('Worker version: {}.'.format(VERSION))
        self.__job_executor: [IExecutor] = None

    def slave_forever(self):
        # set up listening port
        listener = Serve(net_type='fcnet')
        try:
            while True:
                self.client_logger.log_message('Worker started with network type \'FCNet\'.')
                try:
                    # com : Communication_Controller
                    with listener.acquire() as com:
                        self.client_logger.log_message(
                            'Job submission received. Node assigned node_id({})'.format(com.Node_Id))

                        self.dispatch(com)

                        self.client_logger.log_message('Current session closed, node_id({}).'.format(com.Node_Id))
                        self.client_logger.log_message('Worker restarting...')
                        time.sleep(1)
                except OSError:
                    self.client_logger.log_message("Initialization server exited without report.")
                except ConnectionResetError:
                    self.client_logger.log_message("Initialization server exited without report.")

        except KeyboardInterrupt:
            self.client_logger.log_error('Worker shutdown by interruption.')
            listener.close()

    @staticmethod
    def __recv_pack(com: ICommunication_Controller, timeout: int = 100):
        data = None
        id_from = None
        time_out_end = time.time() + timeout
        # requests with timeout check
        while data is None:
            id_from, data = com.get_one(blocking=False)
            time.sleep(0.01)
            # Assertion, this node count as one
            assert Initialization_Server in com.available_clients, "Initialization server exited without finishing the initialization."
            assert time.time() < time_out_end, "Maximum waiting time exceed."
        return id_from, data

    def dispatch(self, com: ICommunication_Controller):
        """
            Get first package and find out what to do.
            All exceptions will be handled here, and trace back information will
            be recorded to client_logger.
            Used job_submit.py --retrieve to get trace back log.
        :param com:
        :return:
        """
        results = None
        try:
            id_from = com.Node_Id  # Node_Id 本机的节点id
            req = None

            while id_from != Initialization_Server:  # Initialization_Server用于判断是否收到数据
                # 返回值：id_from, 01 = com.get_one(blocking=False)
                id_from, req = Worker.__recv_pack(com, Init_Job_Submission_Timeout_Limit_Sec)
            # isinstance()，判断是否为某个类的对象，
            # req实例，参数2：直接或间接类名、基本类型或者由它们组成的元组
            if isinstance(req, SubmitJob):
                # Report Version， 往集群发送版本号
                com.send_one(Initialization_Server, Version(node_id=com.Node_Id))
                self.client_logger.log_message('ACK job submission.')
                if self.initialize(com, req):
                    results = self.do_training(com)

            if isinstance(req, RequestWorkingLog):
                self.client_logger.log_message('ACK logfile reclaim.')

        except Exception as e:
            # print DEBUG message
            import sys
            import traceback
            exc_type, exc_value, exc_tb = sys.exc_info()
            exc_tb = traceback.format_exception(exc_type, exc_value, exc_tb)
            exc_format = "".join(exc_tb)
            self.client_logger.log_error('Exception occurred: {}\n\t{}'.format(e, exc_format))
            # print DEBUG message

        self.post_log(com, results)

    def post_log(self, com: ICommunication_Controller, other_contents: object):
        """
            Post worker log file to coordinator.
        :param other_contents: other content can be attached
        :param com:
        :return:
        """
        posting_files = [self.client_logger.File_Name]
        if isinstance(self.__job_executor, AbsExecutor):
            for filename in self.__job_executor.trace_files():
                posting_files.append(filename)

        # Post files
        com.send_one(Initialization_Server, DoneType(com.Node_Id, posting_files, other_contents))

    def initialize(self, com: ICommunication_Controller, job_info: SubmitJob) -> bool:
        """
            初始化执行环境
            Initialize execution environment
        :param com: Communication process
        :param job_info: job info
        :return:
        """
        # restoring 01
        job_info.restore()
        # get info
        ready_state = set()
        total_nodes = job_info.work_group
        eta_waiting_time = job_info.waiting_time

        self.__job_executor: AbsExecutor = job_info.executioner(com.Node_Id, job_info.work_group)  # 获取提交者下发的执行器，初始化PSGDWorkerExecutor

        # Acknowledge requests
        requests = self.__job_executor.requests()  # req.setting, req.model,...
        replies = []
        # Ask for replies
        for req in requests:
            com.send_one(Initialization_Server, RequestPackage(req))

        req_format = "\tRequests List:\n\t\t--> {}".format("\n\t\t--> ".join([str(req) for req in requests]))
        self.client_logger.log_message('Request 01: ({})\n{}'.format(len(requests), req_format))
        self.client_logger.log_message('ETA: ({})'.format(eta_waiting_time))
        # Set job executor to ready state
        while not self.__job_executor.ready():

            id_from, data = Worker.__recv_pack(com, eta_waiting_time)

            self.client_logger.log_message('Ack package, type: ({})'.format(data.__class__.__name__))
            # restoring 01
            if isinstance(data, IReplyPackage):
                data.restore()
                replies.append(data)

                if len(replies) == len(requests):
                    requests = self.__job_executor.satisfy(replies)
                    for req in requests:
                        com.send_one(Initialization_Server, RequestPackage(req))
                    self.client_logger.log_message('Request 01: ({}).'.format(requests))
                    self.client_logger.log_message('ETA: ({})'.format(eta_waiting_time))
                    replies.clear()

            # pass to sync
            elif isinstance(data, ReadyType):
                ready_state = ready_state | data.current_ready()

        self.client_logger.log_message('Submit stage complete, Total bytes sent: {}'.format(com.Com.bytes_sent))
        self.client_logger.log_message('Submit stage complete, Total bytes read: {}'.format(com.Com.bytes_read))

        self.client_logger.log_message('Synchronize timeline with cluster.')

        Worker.synchronize(com, ready_state, total_nodes, eta_waiting_time)

        return True

    @staticmethod
    def synchronize(com: ICommunication_Controller, ready_state: set, total_nodes: set, timeout: int):
        """
            Synchronize timeline with cluster.
            Make sure all nodes exits this method with same time.
        :param com: communication controller
        :param ready_state: how much nodes is ready now
        :param total_nodes: how much nodes we need for the job
        :param timeout: timeout limit in seconds, vaguely accuracy
        :return:
        """
        dead_line = time.time() + timeout

        ready_state.add(com.Node_Id)
        for id in com.available_clients:
            com.send_one(id, ReadyType(ready_state))

        while ready_state & total_nodes != total_nodes:
            assert time.time() < dead_line, "Maximum waiting time exceed."

            current_active = set(com.available_clients) | {com.Node_Id}
            assert current_active & total_nodes == total_nodes, \
                "Current nodes: {}, required nodes: {}.".format(current_active, total_nodes)
            # inc time clock
            time.sleep(0.01)

            # check ready state
            id_from, data = com.get_one(blocking=False)
            if isinstance(data, ReadyType):
                ready_state = ready_state | data.current_ready()

    def do_training(self, com: ICommunication_Controller) -> object:
        """
            Execute job.
        """
        self.client_logger.log_message('Execution process started.')
        begin = time.time()
        result = self.__job_executor.start(com)
        end = time.time()

        self.client_logger.log_message('Execution complete, time:{}'.format(end - begin))
        self.client_logger.log_message('Execution stage complete, Total bytes sent: {}'.format(com.Com.bytes_sent))
        self.client_logger.log_message('Execution stage complete, Total bytes read: {}'.format(com.Com.bytes_read))
        self.client_logger.log_message('Execution process exited.')

        return result
