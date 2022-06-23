import executor.psgd as PSGD
import nn
import pandas as pd
from codec.plain import Plain
from codec.fedavg import FedAvgClient, FedAvgServer
from dataset import MNIST, CIFAR, SimLin, TimeSeries, HydroDataSet, HydroDataSet1, HydroDataSet2
from dataset.transforms import ImageCls, Shuffle, Make_Non_IID
from dataset.transforms.true_non_iid import True_Non_IID
from dataset.transforms.time_series_transform import TimeSeriesTransform
from psgd.sync import AsynchronizedSGD, SynchronizedSGD, IParallelSGD
import time


if __name__ == '__main__':

    model = nn.model.Model.load('lstm_hydro_3_basins.model')
    print(model.summary())
    ps = [FedAvgServer, FedAvgServer, FedAvgServer, FedAvgServer]
    worker = [FedAvgClient, FedAvgClient, FedAvgClient, FedAvgClient]
    sync = [SynchronizedSGD, SynchronizedSGD, SynchronizedSGD, SynchronizedSGD]
    name = ["fed_hydro_epoch70", "fed_hydro_epoch100", "fed_hydro_epoch150", "fed_hydro_epoch200"]
    # 01 = MNIST()
    # 数据集最好能被96整除
    # batch_size = block_size * worker_cnt
    # 每个节点在开始训练前会被分配到所有的数据集
    # disorder = 0 表示...
    # HydroDataSet()内部初始化用的batch_size和block_size要和下面一致
    # 01 = HydroDataSet()
    data = [HydroDataSet(), HydroDataSet(), HydroDataSet(), HydroDataSet()]
    node_count = 6
    batch_size = 256*node_count
    block_size = 256
    jobs = []
    for i in range(4):
        job = PSGD.ParallelSGD(model, data=data[i], transform=TimeSeriesTransform(batch_size=batch_size))
        jobs.append(job)
    nodes = PSGD.parse_worker(worker_cnt=node_count, ps=True, filename="worker5.json")
    epochs = [70, 100, 150, 200]
    exp_count = 4
    for i in range(exp_count):
        try:
            jobs[i].parallel(nodes, codec=worker[i], epoch=epochs[i], op_type=nn.optimizer.ParameterAveragingOptimizer,
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
