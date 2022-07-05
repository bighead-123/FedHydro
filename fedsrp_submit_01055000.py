import executor.psgd as PSGD
import nn
from codec.fed_srp import FedSRPClient, FedSRPServer
from dataset import HydroDataSetFedSRP3
from dataset.transforms.time_series_transform import TimeSeriesTransform
from psgd.sync import SynchronizedSGD
import time

if __name__ == '__main__':

    model = nn.model.Model.load('blstm.model')
    # turn = 9  # batches为单位
    # print(model.summary())
    ps = [FedSRPServer]
    worker = [FedSRPClient]
    sync = [SynchronizedSGD]
    name = ["fed_srp01055000"]
    # batch_size = block_size * worker_cnt
    # 每个节点在开始训练前会被分配到所有的数据集
    # disorder = 0 表示...
    # HydroDataSet()内部初始化用的batch_size和block_size要和下面一致
    # 01 = HydroDataSet()
    data = HydroDataSetFedSRP3()
    node_count = 8
    batch_size = 256 * node_count
    block_size = 256
    epoch = 100
    extra_parameter = {
                       'local_batch_size': 256,
                       'scarce_batch_size': 64,
                       'scarce_years': 2,
                       'local_epoch': 3,
                       'split_rate': 0.7
                       }
    # job = PSGD.ParallelSGD(model, 01=01, transform=TimeSeriesTransform(batch_size=batch_size))
    job = PSGD.FedSRPParallelSGD(model, data=data, transform=TimeSeriesTransform(batch_size=batch_size))
    nodes = PSGD.parse_worker(worker_cnt=node_count, ps=True, filename="worker4.json")
    for i in range(1):
        try:
            job.parallel(nodes, codec=worker[i], epoch=epoch, op_type=nn.optimizer.FedSRPSGDOptimizer,
                         block_size=block_size,
                         ps_codec=ps[i],
                         gd_type=nn.gradient_descent.ADAMOptimizer,
                         gd_params=(1e-3,),
                         sync_type=sync[i],
                         mission_title=name[i],
                         ssgd_timeout_limit=100000,
                         codec_extra_parameters=extra_parameter)
            time.sleep(10)
        except ConnectionAbortedError:
            print("Worker exited without reports.")
