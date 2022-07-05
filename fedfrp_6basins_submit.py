import executor.psgd as PSGD
import nn
from codec.fedavg import FedAvgClient, FedAvgServer
from dataset import  HydroDataSet
from dataset.transforms.time_series_transform import TimeSeriesTransform
from psgd.sync import SynchronizedSGD
import time


if __name__ == '__main__':

    model = nn.model.Model.load('lstm_hydro.model')
    print(model.summary())
    ps = [FedAvgServer]
    worker = [FedAvgClient]
    sync = [SynchronizedSGD]
    name = ["fed_hydro_6basins_"]
    # batch_size = block_size * worker_cnt
    # 每个节点在开始训练前会被分配到所有的数据集
    # HydroDataSet()内部初始化用的batch_size和block_size要和下面一致
    data = HydroDataSet()
    batch_size = 256 * 6
    block_size = 256
    job = PSGD.ParallelSGD(model, data=data, transform=TimeSeriesTransform(batch_size=batch_size))
    nodes = PSGD.parse_worker(worker_cnt=6, ps=True, filename="worker.json")
    for i in range(1):
        try:
            job.parallel(nodes, codec=worker[i], epoch=100, op_type=nn.optimizer.ParameterAveragingOptimizer,
                         block_size=block_size,
                         ps_codec=ps[i],
                         gd_type=nn.gradient_descent.ADAMOptimizer,
                         gd_params=(1e-3,),
                         sync_type=sync[i],
                         mission_title=name[i],
                         ssgd_timeout_limit=200000)
            time.sleep(10)
        except ConnectionAbortedError:
            print("Worker exited without reports.")
