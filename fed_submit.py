import executor.psgd as PSGD
import nn
from codec.fedavg import FedAvgClient,FedAvgServer
from dataset import MNIST
from dataset.transforms import ImageCls
from dataset.transforms.true_non_iid import True_Non_IID
from psgd.sync import  SynchronizedSGD
import time

if __name__ == '__main__':
    # 1、搭建模型
    # 2、准备数据集
    # 3、修改提交者
    # ps：
    # worker：
    model = nn.model.Model.load('MNISTDNN.model')
    ps = [FedAvgServer]
    worker = [FedAvgClient]  # 编码器
    sync = [SynchronizedSGD]  # AsynchronizedSGD
    name = ["fed_test"]
    data = MNIST()
    # transform：预处理操作
    job = PSGD.ParallelSGD(model, data=MNIST(), transform=True_Non_IID(batch_size=64*8, disorder=0.3).add(ImageCls()))
    # worker_count
    nodes = PSGD.parse_worker(worker_cnt=2, ps=True, filename="worker2.json")
    for i in range(1):
        try:
            job.parallel(nodes, codec=worker[i], epoch=20, op_type=nn.optimizer.ParameterAveragingOptimizer,
                         ps_codec=ps[i],
                         gd_type=nn.gradient_descent.ADAMOptimizer,
                         gd_params=(3e-5,),
                         sync_type=sync[i],
                         mission_title=name[i],
                         ssgd_timeout_limit=20000)
            time.sleep(10)
        except ConnectionAbortedError:
            print("Worker exited without reports.")
