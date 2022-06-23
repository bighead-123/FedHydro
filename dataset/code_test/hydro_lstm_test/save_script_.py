from dataset.code_test.utils_test.get_hydro_data import GetHydroData
import nn
from pylab import mpl, text
from nn.utils.nse_util import calc_nse
from nn.utils.rmse_util import cal_rmse
from nn.utils.mae_util import cal_mae
from utils.constants import model_path, model_path2
from dataset.series_data.utils.generate_data import get_basin_ids
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
from dataset.series_data.utils.feature_fusion import feature_fusion

hidden_size = 20  # Number of LSTM cells
dropout_rate = 0  # Dropout rate of the final fully connected Layer [0.0, 1.0]
learning_rate = 0.001  # Learning rate used to update the weights
sequence_length = 30  # Length of the meteorological record provided to the network
batch_size = 256


# 单机测试记录
def single_model_test(model_name: str, basin_id):
    getHydroData = GetHydroData(basin_id, sequence_length)
    train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()
    model = nn.model.Model.load(model_name)
    predict_y = model.predict(test_x)
    ds_test = getHydroData.get_ds_test()
    predict_y = ds_test.local_rescale(predict_y, variable='output')
    # 计算NSE
    nse = calc_nse(test_y, predict_y)
    # 计算RMSE
    rmse = cal_rmse(test_y, predict_y)
    mae = cal_mae(test_y, predict_y)
    return nse, rmse, mae


# 全局模型测试记录
def fed_model_test(global_model_name, basin_id):
    getHydroData = GetHydroData(basin_id, sequence_length)
    train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()
    model = nn.model.Model.load(global_model_name)
    predict_y = model.predict(test_x)
    ds_test = getHydroData.get_ds_test()
    predict_y = ds_test.local_rescale(predict_y, variable='output')
    # 计算NSE
    nse = calc_nse(test_y, predict_y)
    # 计算RMSE
    rmse = cal_rmse(test_y, predict_y)
    mae = cal_mae(test_y, predict_y)
    return nse, rmse, mae


# 模型再测试记录
def model_retrain(global_model_name, basin_id, epoch):
    getHydroData = GetHydroData(basin_id, sequence_length)
    train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()
    print(train_x[0])
    model = nn.model.Model.load(global_model_name)
    model.compile(nn.gradient_descent.ADAMOptimizer(alpha=learning_rate))
    # 微调全局模型
    model.fit(train_x, train_y, epoch=epoch, batch_size=batch_size)
    ds_test = getHydroData.get_ds_test()
    predict_y = model.predict(test_x)
    predict_y = ds_test.local_rescale(predict_y, variable='output')

    # 计算NSE
    nse = calc_nse(test_y, predict_y)
    # 计算RMSE
    rmse = cal_rmse(test_y, predict_y)
    mae = cal_mae(test_y, predict_y)
    return nse, rmse, mae


def model_few_shot_retrain(global_model_name, basin_id, epoch):
    getHydroData = GetHydroData(basin_id, sequence_length)
    train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()
    print(train_x[0])
    model = nn.model.Model.load(global_model_name)
    model.compile(nn.gradient_descent.ADAMOptimizer(alpha=learning_rate))
    train_x = train_x[:500]
    train_y = train_y[:500]
    test_x = test_x[:50]
    test_y = test_y[:50]
    # 微调全局模型
    model.fit(train_x, train_y, epoch=epoch, batch_size=32)
    ds_test = getHydroData.get_ds_test()
    predict_y = model.predict(test_x)
    predict_y = ds_test.local_rescale(predict_y, variable='output')

    plt.plot(list(range(len(test_y))), test_y, label='Observed')
    plt.plot(list(range(len(predict_y))), predict_y, label='Prediction')
    plt.legend()
    plt.show()
    # 计算NSE
    nse = calc_nse(test_y, predict_y)
    # 计算RMSE
    rmse = cal_rmse(test_y, predict_y)
    mae = cal_mae(test_y, predict_y)
    return nse, rmse, mae


# 模型再测试记录
def gen_retrain_model(global_model_name, basin_id, epoch):
    getHydroData = GetHydroData(basin_id, sequence_length)
    train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()
    model = nn.model.Model.load(global_model_name)
    model.compile(nn.gradient_descent.ADAMOptimizer(alpha=learning_rate))
    # 微调全局模型
    model.fit(train_x, train_y, epoch=epoch, batch_size=batch_size)
    model.save('fed_model_retrain/fed_model/MODEL-fed_hydro_6basins-N(0)'+str(basin_id)+'_retrain10.model')


# 将测试记录保存成csv格式
def save_csv(basin_ids, save_path, global_model_name):
    # 1. 创建文件对象
    f = open(save_path, 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)

    # 写入列表头
    csv_writer.writerow(["Basin Id", "  ", "NSE", "RMSE", "MAE"])
    for basin_id in basin_ids:
        single_model_name = 'D:/河海大学/研究课题/水文预报/课题实验/Parallel-SGD/dataset/code_test/' \
                            'single_model/lstm_hydro_'+str(basin_id)+'.model'

        csv_writer.writerow([str(basin_id), "", "", "", ""])

        nse1, rmse1, mae1 = single_model_test(single_model_name, basin_id)
        csv_writer.writerow(["", "single model", str(nse1), str(rmse1), str(mae1)])

        nse2, rmse2, mae2 = fed_model_test(global_model_name, basin_id)
        csv_writer.writerow(["", "federated model", str(nse2), str(rmse2), str(mae2)])

        nse3, rmse3, mae3 = model_retrain(global_model_name, basin_id, 10)
        csv_writer.writerow(["", "federated model retrain after 10 epoch", str(nse3), str(rmse3), str(mae3)])

        nse4, rmse4, mae4 = model_retrain(global_model_name, basin_id, 20)
        csv_writer.writerow(["", "federated model retrain after 20 epoch", str(nse4), str(rmse4), str(mae4)])


def gen_fig_res(basin_ids, save_path, global_model_name):
    for basin_id in basin_ids:
        getHydroData = GetHydroData(basin_id, sequence_length)
        train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()
        single_lstm_model = nn.model.Model.load('single_lstm_model/lstm_hydro_01030500.model')
        fed_lstm_model = nn.model.Model.load('fed_model/MODEL-fed_hydro_6basins-N(0).model')
        fed_lstm_model_10 = nn.model.Model.load('fed_model/MODEL-fed_hydro_6basins-N(0)_'+str(basin_id)+'_retrain10.model')
        ds_test = getHydroData.get_ds_test()
        # lstm model
        single_predict_y = single_lstm_model.predict(test_x)
        single_predict_y = ds_test.local_rescale(single_predict_y, variable='output')
        single_nse = calc_nse(test_y, single_predict_y)  # 计算NSE
        test_y = test_y[:1000]
        single_predict_y = single_predict_y[:1000]
        test_y = ds_test.reshape_discharge(test_y)
        single_predict_y = ds_test.reshape_discharge(single_predict_y)

        # fed_lstm
        fed_predict_y = fed_lstm_model.predict(test_x)
        fed_predict_y = ds_test.local_rescale(fed_predict_y, variable='output')
        fed_nse = calc_nse(test_y, fed_predict_y)  # 计算NSE
        test_y = test_y[:1000]
        fed_predict_y = fed_predict_y[:1000]
        test_y = ds_test.reshape_discharge(test_y)
        fed_predict_y = ds_test.reshape_discharge(fed_predict_y)

        # fed-lstm-10
        fed10_predict_y = fed_lstm_model_10.predict(test_x)
        fed10_predict_y = ds_test.local_rescale(fed10_predict_y, variable='output')
        fed10_nse = calc_nse(test_y, fed10_predict_y)  # 计算NSE
        test_y = test_y[:1000]
        fed10_predict_y = fed10_predict_y[:1000]
        test_y = ds_test.reshape_discharge(test_y)
        fed10_predict_y = ds_test.reshape_discharge(fed10_predict_y)

        print("model-worker0,NSE:", single_nse)
        print("model-worker0,fed_NSE:", fed_nse)
        print("model-worker0,fed10_NSE:", fed10_nse)

        # plt.subplot(3, 1, 1)  # 2行1列，第一张图
        plt.plot(list(range(len(test_y))), test_y, color='b', label='Observed')
        plt.plot(list(range(len(single_predict_y))), single_predict_y, color='orange', label='LSTM Model')
        plt.plot(list(range(len(fed_predict_y))), fed_predict_y, color='green', label='Fed-LSTM')
        plt.plot(list(range(len(fed10_predict_y))), fed10_predict_y, color='blue', label='Fed-LSTM-10')
        # plt.title(f"basin:{basin_id}, NSE:{nse:.3f}")
        plt.xlabel('day number')
        plt.ylabel('discharge')
        plt.legend()
        plt.savefig('fig_res/' + str(basin_id) + '.svg', format='svg')
        plt.show()


def gen_single_basin_pred(basin_id):
    config = {
        "font.family": 'serif',
        # "font.size": 12,
        "mathtext.fontset": 'stix',
        # "font.serif": ['SimSun'],
    }
    plt.rcParams.update(config)

    path = 'D:/河海大学/研究课题/研究课题/实验相关/PSGD/Parallel-SGD/dataset/code_test/hydro_lstm_test/'
    getHydroData = GetHydroData(basin_id, sequence_length)
    train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()
    single_lstm_model = nn.model.Model.load(path + 'single_lstm_model/lstm_hydro_' + str(basin_id) + '.model')
    fed_lstm_model = nn.model.Model.load(path + 'fed_model/MODEL-fed_hydro_6basins-N(0).model')
    fed_lstm_model_10 = nn.model.Model.load(
        path + 'fed_model_retrain/MODEL-fed_hydro_6basins-N(0)_' + str(basin_id) + '_retrain10.model')
    ds_test = getHydroData.get_ds_test()
    # lstm model
    single_predict_y = single_lstm_model.predict(test_x)
    single_predict_y = ds_test.local_rescale(single_predict_y, variable='output')
    single_nse = calc_nse(test_y, single_predict_y)  # 计算NSE
    # test_y = test_y[:1000]
    # single_predict_y = single_predict_y[:1000]
    # test_y = ds_test.reshape_discharge(test_y)
    # single_predict_y = ds_test.reshape_discharge(single_predict_y)

    # fed_lstm
    fed_predict_y = fed_lstm_model.predict(test_x)
    fed_predict_y = ds_test.local_rescale(fed_predict_y, variable='output')
    fed_nse = calc_nse(test_y, fed_predict_y)  # 计算NSE
    # test_y = test_y[:1000]
    # fed_predict_y = fed_predict_y[:1000]
    # test_y = ds_test.reshape_discharge(test_y)
    # fed_predict_y = ds_test.reshape_discharge(fed_predict_y)

    # fed-lstm-10
    fed10_predict_y = fed_lstm_model_10.predict(test_x)
    fed10_predict_y = ds_test.local_rescale(fed10_predict_y, variable='output')
    fed10_nse = calc_nse(test_y, fed10_predict_y)  # 计算NSE
    # test_y = test_y[:1000]
    # fed10_predict_y = fed10_predict_y[:1000]
    # test_y = ds_test.reshape_discharge(test_y)
    # fed10_predict_y = ds_test.reshape_discharge(fed10_predict_y)

    # test_y = ds_test.reshape_discharge(test_y)
    plt.rcParams['figure.figsize'] = (8.0, 6.0)
    # plt.rcParams['figure.dpi'] = 120
    # plt.rcParams['savefig.dpi'] = 120
    plt.subplot(3, 1, 1)  # 3行1列，第一张图
    plt.plot(list(range(len(single_predict_y))), single_predict_y, color='orange', label='LSTM-Model')
    plt.plot(list(range(len(test_y))), test_y, color='b', label='Actual')
    # plt.title(f"basin:{basin_id}, NSE:{single_nse:.3f}")
    # plt.xlabel('day number')
    plt.ylabel('discharge')
    plt.legend(frameon=True, loc="upper left", fontsize='x-small')
    # plt.savefig(path+'/fig_res/' + str(basin_id) + '_lstm.svg', format='svg')

    plt.subplot(3, 1, 2)  # 3行1列，第2张图
    plt.plot(list(range(len(fed_predict_y))), fed_predict_y, color='green', label='Fed-LSTM')
    plt.plot(list(range(len(test_y))), test_y, color='b', label='Actual')
    # plt.title(f"basin:{basin_id}, NSE:{fed_nse:.3f}")
    # plt.xlabel('day number')
    plt.ylabel('discharge')
    plt.legend(frameon=True, loc="upper left", fontsize='x-small')
    # plt.savefig(path+'/fig_res/' + str(basin_id) + '_fed_lstm.svg', format='svg')

    plt.subplot(3, 1, 3)  # 3行1列，第3张图
    plt.plot(list(range(len(fed10_predict_y))), fed10_predict_y, color='r', label='Fed-LSTM-10')
    plt.plot(list(range(len(test_y))), test_y, color='b', label='ACtual')
    # plt.title(f"basin:{basin_id}, NSE:{fed10_nse:.3f}")
    plt.xlabel('day number')
    plt.ylabel('discharge')
    plt.legend(frameon=True, loc="upper left", fontsize='x-small')
    plt.savefig(path+'/fig_res/' + str(basin_id) + '_model_compare.svg', format='svg')
    plt.show()


# 绘制Fed-Hydro在N流域上的预测效果图
def gen_fed_hydro_two_basin_pred(basin_ids):
    # config = {
    #     # "font.family": 'Times New Roman',
    #     # "font.size": 12,
    #     "mathtext.fontset": 'SimSun',
    #     # "font.serif": ['SimSun'],
    # }
    # plt.rcParams.update(config)

    path = 'D:/河海大学/研究课题/研究课题/实验相关/PSGD/Parallel-SGD/dataset/code_test/hydro_lstm_test/fed_model_retrain/'

    # 流域1
    getHydroData1 = GetHydroData(basin_ids[0], sequence_length)
    train_x, train_y, val_x, val_y, test_x1, test_y1 = getHydroData1.get_data()
    fed_lstm_model1 = nn.model.Model.load(path + 'MODEL-fed_hydro_6basins-N(0)_'+str(basin_ids[0])+'_retrain10.model')
    ds_test1 = getHydroData1.get_ds_test()
    # fed_lstm
    fed_predict_y1 = fed_lstm_model1.predict(test_x1)
    fed_predict_y1 = ds_test1.local_rescale(fed_predict_y1, variable='output')

    # 流域2
    getHydroData2 = GetHydroData(basin_ids[1], sequence_length)
    train_x, train_y, val_x, val_y, test_x2, test_y2 = getHydroData2.get_data()
    fed_lstm_model2 = nn.model.Model.load(path + 'MODEL-fed_hydro_6basins-N(0)_'+str(basin_ids[1])+'_retrain10.model')
    ds_test2 = getHydroData1.get_ds_test()
    # fed_lstm
    fed_predict_y2 = fed_lstm_model2.predict(test_x2)
    fed_predict_y2 = ds_test2.local_rescale(fed_predict_y2, variable='output')

    start_date = ds_test1.dates[1] - pd.DateOffset(364)
    end_date = ds_test1.dates[1]
    date_range = pd.date_range(start_date, end_date)
    plt.rcParams['figure.figsize'] = (16.0, 6)
    # plt.figure(figsize=(8, 4), dpi=80)
    ax1 = plt.subplot(1, 2, 1)  # 3行1列，第2张图
    # plt.figure(figsize=(8, 4), dpi=80)
    plt.plot(date_range, fed_predict_y1[-365:], color='red', label='Prediction')
    plt.plot(date_range, test_y1[-365:], color='b', label='Observation')
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    # plt.title(f"basin:{basin_id}, NSE:{fed_nse:.3f}")

    plt.xlabel('Date', fontdict={'family': 'Times New Roman', 'size': 16})
    # text('流量', fontdict={'family': 'SimSun', 'size': 12})
    plt.ylabel('Discharge mm/day', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.xticks(size=14)
    plt.yticks(size=14)

    # plt.legend(frameon=True, loc="upper left", fontsize='x-small')
    plt.legend(frameon=True, loc="upper left")
    plt.title('流域1', fontdict={'family': 'SimSun', 'size': 16})
    # plt.title('流域3', fontdict={'family': 'SimSun', 'size': 12})
    # plt.title('流域5', fontdict={'family': 'SimSun', 'size': 12})
    # plt.savefig(path+'/fig_res/' + str(basin_id) + '_fed_lstm.svg', format='svg')

    ax2 = plt.subplot(1, 2, 2)  # 3行1列，第3张图
    plt.plot(date_range, fed_predict_y2[-365:], color='red', label='Prediction')
    plt.plot(date_range, test_y2[-365:], color='b', label='Observation')
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    plt.xlabel('Date', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.xticks(size=14)
    plt.yticks(size=14)
    # plt.ylabel('discharge mm/day', fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.legend(frameon=True, loc="upper left", fontsize='x-small')
    plt.legend(frameon=True, loc="upper left")
    plt.title('流域2', fontdict={'family': 'SimSun', 'size': 16})
    # plt.title('流域4', fontdict={'family': 'SimSun', 'size': 12})
    # plt.title('流域6', fontdict={'family': 'SimSun', 'size': 12})
    # plt.savefig(path+'/fig_res/' + str(basin_id) + '_model_compare.svg', format='svg')
    plt.show()


# 生成不同流域数量，不同本地epoch联邦模型的NSE
def gen_n_basins_fed_model_table(global_fed_model, scene_num, in_basin_ids, without_basin_ids,
                                 local_N1, local_N2, local_N3):
    save_path = "D:\\河海大学\\研究课题\\研究课题\\实验相关\\PSGD\\Parallel-SGD\\dataset\\code_test\\" \
                "hydro_lstm_test\\csv_res\\n_basins_fed_model_res_51015.csv"

    # 1. 创建文件对象
    f = open(save_path, 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)
    i = 1
    # 写入列表头
    csv_writer.writerow(["Scene", "Model", "In Train", "Without Train", "In RMSE", "Without RMSE", "In MAE", "Without MAE"])
    model_name = global_fed_model
    scene_num = scene_num // 2
    # in
    in_fed_model_5_list = []
    in_fed_model_10_list = []
    in_fed_model_20_list = []

    in_fedrmse_model_5_list = []
    in_fedrmse_model_10_list = []
    in_fedrmse_model_20_list = []

    in_fedmae_model_5_list = []
    in_fedmae_model_10_list = []
    in_fedmae_model_20_list = []

    # without
    without_fed_model_5_list = []
    without_fed_model_10_list = []
    without_fed_model_20_list = []

    without_fedrmse_model_5_list = []
    without_fedrmse_model_10_list = []
    without_fedrmse_model_20_list = []

    without_fedmae_model_5_list = []
    without_fedmae_model_10_list = []
    without_fedmae_model_20_list = []

    for num in range(1, scene_num+1):
        global_fed_model = model_name + "MODEL-fed_hydro_" + str(3*num)+"basins_t30_210-N(0).model"
        in_nse_list5 = []
        in_rmse_list5 = []
        in_mae_list5 = []

        without_nse_list5 = []
        without_rmse_list5 = []
        without_mae_list5 = []

        in_nse_list10 = []
        in_rmse_list10 = []
        in_mae_list10 = []

        without_nse_list10 = []
        without_rmse_list10 = []
        without_mae_list10 = []

        in_nse_list20 = []
        in_rmse_list20 = []
        in_mae_list20 = []

        without_nse_list20 = []
        without_rmse_list20 = []
        without_mae_list20 = []

        # 记录参与训练的流域评估NSE
        for basin_id in in_basin_ids:
            nse1, rmse1, mae1 = model_retrain(global_fed_model, basin_id, local_N1)
            nse2, rmse2, mae2 = model_retrain(global_fed_model, basin_id, local_N2)
            nse3, rmse3, mae3 = model_retrain(global_fed_model, basin_id, local_N3)
            in_nse_list5.append(nse1)
            in_nse_list10.append(nse2)
            in_nse_list20.append(nse3)

            in_rmse_list5.append(rmse1)
            in_rmse_list10.append(rmse2)
            in_rmse_list20.append(rmse3)

            in_mae_list5.append(mae1)
            in_mae_list10.append(mae2)
            in_mae_list20.append(mae3)
        # 计算平均nse
        in_nse_avg5 = np.mean(in_nse_list5)
        in_nse_avg10 = np.mean(in_nse_list10)
        in_nse_avg20 = np.mean(in_nse_list20)

        # 计算平均rmse
        in_rmse_avg5 = np.mean(in_rmse_list5)
        in_rmse_avg10 = np.mean(in_rmse_list10)
        in_rmse_avg20 = np.mean(in_rmse_list20)

        # 计算平均mae
        in_mae_avg5 = np.mean(in_mae_list5)
        in_mae_avg10 = np.mean(in_mae_list10)
        in_mae_avg20 = np.mean(in_mae_list20)

        # 添加到返回列表
        in_fed_model_5_list.append(in_nse_avg5)
        in_fed_model_10_list.append(in_nse_avg10)
        in_fed_model_20_list.append(in_nse_avg20)

        in_fedrmse_model_5_list.append(in_rmse_avg5)
        in_fedrmse_model_10_list.append(in_rmse_avg10)
        in_fedrmse_model_20_list.append(in_rmse_avg20)

        in_fedmae_model_5_list.append(in_mae_avg5)
        in_fedmae_model_10_list.append(in_mae_avg10)
        in_fedmae_model_20_list.append(in_mae_avg20)

        # 记录未参与训练的流域评估NSE
        for basin_id in without_basin_ids:
            nse1, rmse1, mae1 = model_retrain(global_fed_model, basin_id, local_N1)
            nse2, rmse2, mae2 = model_retrain(global_fed_model, basin_id, local_N2)
            nse3, rmse3, mae3 = model_retrain(global_fed_model, basin_id, local_N3)
            without_nse_list5.append(nse1)
            without_nse_list10.append(nse2)
            without_nse_list20.append(nse3)

            without_rmse_list5.append(rmse1)
            without_rmse_list10.append(rmse2)
            without_rmse_list20.append(rmse3)

            without_mae_list5.append(mae1)
            without_mae_list10.append(mae2)
            without_mae_list20.append(mae3)
        # 计算平均nse
        without_nse_avg5 = np.mean(without_nse_list5)
        without_nse_avg10 = np.mean(without_nse_list10)
        without_nse_avg20 = np.mean(without_nse_list20)

        without_rmse_avg5 = np.mean(without_rmse_list5)
        without_rmse_avg10 = np.mean(without_rmse_list10)
        without_rmse_avg20 = np.mean(without_rmse_list20)

        without_mae_avg5 = np.mean(without_mae_list5)
        without_mae_avg10 = np.mean(without_mae_list10)
        without_mae_avg20 = np.mean(without_mae_list20)

        # 添加到返回列表
        without_fed_model_5_list.append(without_nse_avg5)
        without_fed_model_10_list.append(without_nse_avg10)
        without_fed_model_20_list.append(without_nse_avg20)

        without_fedrmse_model_5_list.append(without_rmse_avg5)
        without_fedrmse_model_10_list.append(without_rmse_avg10)
        without_fedrmse_model_20_list.append(without_rmse_avg20)

        without_fedmae_model_5_list.append(without_mae_avg5)
        without_fedmae_model_10_list.append(without_mae_avg10)
        without_fedmae_model_20_list.append(without_mae_avg20)

        sce_no = num*3
        csv_writer.writerow([str(sce_no-2), "Fed-LSTM-"+str(local_N1), str(in_nse_avg5), str(without_nse_avg5), str(in_rmse_avg5), str(without_rmse_avg5), str(in_mae_avg5), str(without_mae_avg5)])
        csv_writer.writerow([str(sce_no-1), "Fed-LSTM-"+str(local_N2), str(in_nse_avg10), str(without_nse_avg10), str(in_rmse_avg10), str(without_rmse_avg10), str(in_mae_avg10), str(without_mae_avg10)])
        csv_writer.writerow([str(sce_no), "Fed-LSTM-"+str(local_N3), str(in_nse_avg20), str(without_nse_avg20), str(in_rmse_avg20), str(without_rmse_avg20), str(in_mae_avg20), str(without_mae_avg20)])
    return in_fed_model_5_list, in_fed_model_10_list, in_fed_model_20_list, \
            without_fed_model_5_list, without_fed_model_10_list, without_fed_model_20_list


def get_3basins_local_1(in_basin_ids, without_basin_ids, local_N1):
    global_fed_model = "./MODEL-fed_hydro_3basins_t30_210-N(0).model"
    in_nse_list5 = []
    without_nse_list5 = []
    # 记录参与训练的流域评估NSE
    for basin_id in in_basin_ids:
        nse1, _, _ = model_retrain(global_fed_model, basin_id, local_N1)
        in_nse_list5.append(nse1)
    # 计算平均nse
    in_nse_avg5 = np.mean(in_nse_list5)
    print("nse1:",in_nse_avg5)

    # 记录未参与训练的流域评估NSE
    for basin_id in without_basin_ids:
        nse1, _, _ = model_retrain(global_fed_model, basin_id, local_N1)
        without_nse_list5.append(nse1)
    # 计算平均nse
    without_nse_avg5 = np.mean(without_nse_list5)
    print("nse2", without_nse_avg5)


# 第三章实验3
def gen_n_basins_fed_model_bar_chart(num_list1, num_list2, num_list3, flag=1):
    # config = {
    #     "mathtext.fontset": 'stix'
    # }
    # plt.rcParams.update(config)
    label_list = ['3', '6', '9', '12']  # 横坐标刻度显示值
    fed_model_5 = []
    fed_model_10 = []
    fed_model_20 = []
    x = range(len(num_list1))
    plt.figure(figsize=(11, 8), dpi=60)
    """
    绘制条形图
    left:长条形中点横坐标
    height:长条形高度
    width:长条形宽度，默认值0.8
    label:为后面设置legend准备
    """

    # color4 = ()
    color1 = (255/255, 255/255, 0/255)
    color2 = (233/255, 120/255, 36/255)
    color3 = (0 / 255, 67 / 255, 147 / 255)

    width_bar = 0.15
    rects1 = plt.bar(x, height=num_list1, width=width_bar, alpha=0.8,   color="darkgray", label="FedFRP-5")
    rects2 = plt.bar([i + width_bar+0.05 for i in x], height=num_list2, width=width_bar,  color=color2,  label="FedFRP-10")
    rects3 = plt.bar([i + width_bar*2+0.1 for i in x], height=num_list3, width=width_bar, color="#6488ea", label="FedFRP-15")
    size = 30
    if flag == 1:
        plt.ylim(0, 1)  # y轴取值范围
    # plt.xlabel('流域个数', fontdict={'family': 'SimSun', 'size': 18})
    plt.xlabel('Basin number', fontdict={'family': 'Times New Roman', 'size': size})
    if flag == 1:
        plt.ylabel('NSE', fontdict={'family' : 'Times New Roman', 'size': size})
    elif flag == 2:
        plt.ylabel('RMSE', fontdict={'family': 'Times New Roman', 'size': size})
    else:
        plt.ylabel('MAE', fontdict={'family': 'Times New Roman', 'size': size})
    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """
    plt.xticks([index + width_bar +0.05  for index in x], label_list, FontProperties='Times New Roman', size=size)
    plt.yticks(FontProperties='Times New Roman', size=size)
    legend_size = 24
    # plt.title("")
    if flag == 1:
        plt.legend(prop={'family': 'Times New Roman', 'size': legend_size}, loc="upper left")  # 设置题注
    else:
        plt.legend(prop={'family': 'Times New Roman', 'size': legend_size})  # 设置题注
    plt.show()


def gen_n_basin_fed_model_bar_chart_2(tick_step=1, group_gap=0.2, bar_gap=0):
    '''
       labels : x轴坐标标签序列
       datas ：数据集，二维列表，要求列表每个元素的长度必须与labels的长度一致
       tick_step ：默认x轴刻度步长为1，通过tick_step可调整x轴刻度步长。
       group_gap : 柱子组与组之间的间隙，最好为正值，否则组与组之间重叠
       bar_gap ：每组柱子之间的空隙，默认为0，每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠
       '''
    labels = ['3', '6', '9', '12']
    datas = [num_list1, num_list2, num_list3]
    # ticks为x轴刻度
    ticks = np.arange(len(labels)) * tick_step
    # group_num为数据的组数，即每组柱子的柱子个数
    group_num = len(datas)
    # group_width为每组柱子的总宽度，group_gap 为柱子组与组之间的间隙。
    group_width = tick_step - group_gap
    # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
    bar_span = group_width / group_num
    # bar_width为每个柱子的实际宽度
    bar_width = bar_span - bar_gap
    # baseline_x为每组柱子第一个柱子的基准x轴位置，随后的柱子依次递增bar_span即可
    baseline_x = ticks - (group_width - bar_span) / 2
    for index, y in enumerate(datas):
        plt.bar(baseline_x + index*bar_span, y, bar_width, alpha=0.8, color="darkgray", label="FedHydro-5")


# 单个模型基于本地数据训练
def single_basin_train(unit_code, basin_id, epoch):
    getHydroData = GetHydroData(basin_id, sequence_length)
    train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()
    single_model = nn.model.SequentialModel.load(model_path)
    single_model.fit_early_stopping(train_x, train_y, val_x, val_y, exp_epoch=50, req_stop_num=5, epoch=epoch, batch_size=256)
    # single_model.save("./single_lstm_model/"+str(unit_code)+"/lstm_hydro_model_unit"+str(unit_code)+"_"+str(basin_id)+".model")
    ds_test = getHydroData.get_ds_test()
    predict_y = single_model.predict(test_x)
    predict_y = ds_test.local_rescale(predict_y, variable='output')
    # 计算NSE
    nse = calc_nse(test_y, predict_y)
    # 计算RMSE
    rmse = cal_rmse(test_y, predict_y)
    mae = cal_mae(test_y, predict_y)
    print("epoch:", epoch, "unit:", unit_code, ",basin:", basin_id, ",nse:", nse)
    return nse, rmse, mae


def cal_nse_rmse_mae(test_y, predict_y):
    """计算nse,rmse,mae"""
    # 计算NSE
    nse = calc_nse(test_y, predict_y)
    # 计算RMSE
    rmse = cal_rmse(test_y, predict_y)
    mae = cal_mae(test_y, predict_y)
    return nse, rmse, mae


def gen_4units_model_result(unit_code, node_count, fed_model_name, single_model_path):
    path = 'D:/河海大学/研究课题/研究课题/实验相关/PSGD/Parallel-SGD/dataset/code_test/hydro_lstm_test/fed_model/'
    # fed_model_name = "./fed_model/MODEL-fed_hydro_unit" + str(unit_code) + "_12basins_t30_210-N(11).model"
    basin_ids = get_basin_ids(unit_code)
    basin_ids = basin_ids[:node_count]
    basin_ids = ['01030500', '01013500', '01031500', '01052500', '01054200', '01055000',
                 '01057000', '01073000', '01078000', '01118300', '01121000', '01123000']
    fed_nse_list = []
    fed_rmse_list = []
    fed_mae_list = []
    # 本地训练
    single_nse_list = []
    single_rmse_list = []
    single_mae_list = []

    # 全局模型
    fed_lstm_nse = []
    fed_lstm_rmse = []
    fed_lstm_mae = []

    model_path = single_model_path + "/" + str(unit_code) + "/"
    fed_lstm = model = nn.model.Model.load(fed_model_name)
    # model.compile(nn.gradient_descent.ADAMOptimizer(alpha=learning_rate))
    for basin_id in basin_ids:
        getHydroData = GetHydroData(basin_id, sequence_length)
        train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()
        ds_test = getHydroData.get_ds_test()
        fed_model_path = fed_model_name
        # fed-hydro-10
        nse1, rmse1, mae1 = model_retrain(fed_model_path, basin_id, epoch=10)
        fed_nse_list.append(nse1)
        fed_rmse_list.append(rmse1)
        fed_mae_list.append(mae1)

        predict_y = fed_lstm.predict(test_x)
        predict_y = ds_test.local_rescale(predict_y, variable='output')
        nse, rmse, mae = cal_nse_rmse_mae(test_y, predict_y)
        fed_lstm_nse.append(nse)
        fed_lstm_rmse.append(rmse)
        fed_lstm_mae.append(mae)
        # 本地训练
        single_model_name = "lstm_hydro_model_unit"+str(unit_code)+"_"+str(basin_id)+".model"
        single_model_path = model_path + single_model_name
        single_model = nn.model.SequentialModel.load(single_model_path)
        predict_y = single_model.predict(test_x)
        predict_y = ds_test.local_rescale(predict_y, variable='output')
        nse2, rmse2, mae2 = cal_nse_rmse_mae(test_y, predict_y)
        single_nse_list.append(nse2)
        single_rmse_list.append(rmse2)
        single_mae_list.append(mae2)

    # added for exp1，小论文
    "nse"
    print("NSE:========================================")
    print("Local_nse", single_nse_list)
    print("FedLSTM_nse", fed_lstm_nse)
    print("FedHydro", fed_nse_list)

    "rmse"
    print("RMSE:========================================")
    print("Local_rmse", single_rmse_list)
    print("FedLSTM_rmse", fed_lstm_rmse)
    print("FedHydro", fed_rmse_list)

    "mae"
    print("MAE:========================================")
    print("Local_nse", single_nse_list)
    print("FedLSTM_nse", fed_lstm_mae)
    print("FedHydro", fed_mae_list)

    nse_list = []
    nse_list.append(fed_nse_list)
    nse_list.append(single_nse_list)

    rmse_list = []
    rmse_list.append(fed_rmse_list)
    rmse_list.append(single_rmse_list)

    mae_list = []
    mae_list.append(fed_mae_list)
    mae_list.append(single_mae_list)

    # 将预测指标保存为csv格式
    save_path = "./csv_res/chapter3_ex4_unit"+str(unit_code)+".csv"  # 多水文单元实验结果

    # 写入列表头
    # name = ["fed_model", "single_model"]
    all_list = []
    all_list.append(nse_list)
    all_list.append(rmse_list)
    all_list.append(mae_list)
    res1 = pd.DataFrame(data=all_list)
    res1.to_csv(save_path)


def gen_unit_single_model(units, node_count, epoch):

    for unit_code in units:
        basin_ids = ['01030500', '01013500', '01031500', '01052500', '01054200', '01055000', '01057000', '01073000', '01078000', '01118300', '01121000', '01123000']
        basin_ids = basin_ids[:node_count]
        for basin_id in basin_ids:
            single_basin_train(unit_code, basin_id, epoch=epoch)


def fusion_feature_test(basin_id):
    getHydroData = GetHydroData(basin_id, sequence_length)
    train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()
    ds_test = getHydroData.get_ds_test()
    weight_list = [0.4, 0.2, 0.2, 0.2, 0.1]
    train_x = feature_fusion(train_x, weight_list)
    test_x = feature_fusion(test_x, weight_list)
    model = nn.model.SequentialModel.load(model_path2)
    model.fit(train_x, train_y, epoch=70, batch_size=256)
    predict_y = model.predict(test_x)
    predict_y = ds_test.local_rescale(predict_y, variable='output')
    nse, rmse, mae = cal_nse_rmse_mae(test_y, predict_y)
    print(nse)


def get_trained_single_model_predict(model_path, basin_id):
    getHydroData = GetHydroData(basin_id, sequence_length)
    train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()
    ds_test = getHydroData.get_ds_test()
    model = nn.model.SequentialModel.load(model_path)
    predict_y = model.predict(test_x)
    predict_y = ds_test.local_rescale(predict_y, variable='output')
    nse, rmse, mae = cal_nse_rmse_mae(test_y, predict_y)
    print(nse)


def get_chapter3_exe4_fig():

    fed_nse1 = [0.8214513641286092, 0.7542472746270342, 0.8199971148188224, 0.7365483856720245, 0.601169267044835,
                0.6810456518023866, 0.7488897677700637, 0.7764051187190378, 0.7712855546506834, 0.7404529698241297,
                0.7519116349099357, 0.7379951455688617]
    single_nse1 = [0.6987040001087441, 0.8082996345999944, 0.7136197200961969, 0.6328089473149567, 0.5857163204641906,
                   0.6945374522522008, 0.6523621610989274, 0.7120794822572174, 0.6666724842910865, 0.6682653051122358,
                   0.7093184389384917, 0.6786574579325968]

    fed_nse2 = [0.5517858170742502, 0.4308757697677188, 0.5433837343437413, 0.6430399349608251, 0.582808534925007, 0.3224952947080141, 0.45641345128180233, 0.22158309621014471, 0.41607589548280166, 0.2381320366231674, 0.4913633818737383, 0.5022140026161825]

    single_nse2 = [0.42503665766684984, 0.37256527991534993, 0.5170274478064845, 0.5409700873887406, 0.4868550919171045, 0.38301472385211255, 0.3243972605867086, 0.2433654177654474, 0.48779269336527675, 0.43349520149537735, 0.4339733496472983, 0.4118860186466481]

    # unit 11
    fed_nse3 = [0.7628535051306087, 0.68537696704412, 0.5072069788268099, 0.6422739428784994, 0.660126426340937, 0.758773592413981, 0.618880752228085, 0.7993294535467049, -0.1926644924743346, 0.5684630908271184, 0.5069437188691708, 0.29396890459928837]


    single_nse3 = [0.729044739987507, 0.583344685294576, 0.47142346791407197, 0.6826625582778016, 0.5950811830346598, 0.6458218000036531, 0.46813992036554763, 0.6747988279561459, 0.3540436830853011, 0.6591659622544086, 0.43612257292357115, 0.43125185494531026]


    # unit 17
    fed_nse4 = [0.5735658651436095, 0.8812296134524112, 0.8728627619027447, 0.49928377903446375, 0.8080258737686972, 0.6987679045305333, 0.8942484993493829, 0.8185967778870663, 0.7505062082946831, 0.7702666137628462, 0.5701215987835464, 0.6678072071943084]

    single_nse4 = [0.569080642021945, 0.8569700675716179, 0.8520942830896174, 0.4310694214232188, 0.7822228396516545, 0.6484386217551941, 0.8949247783317429, 0.8276296625645677, 0.7432401466067821, 0.7512344108079145, 0.3785046940279405, 0.6350464626537651]

    improved_rate_list1 = []
    improved_rate_list2 = []
    improved_rate_list3 = []
    improved_rate_list4 = []

    for i in range(len(fed_nse2)):
        rate1 = (fed_nse1[i] - single_nse1[i]) / single_nse1[i]
        improved_rate_list1.append(rate1)

        rate2 = (fed_nse2[i] - single_nse2[i]) / single_nse2[i]
        improved_rate_list2.append(rate2)

        rate3 = (fed_nse3[i] - single_nse3[i]) / single_nse3[i]
        improved_rate_list3.append(rate3)

        rate4 = (fed_nse4[i] - single_nse4[i]) / single_nse4[i]
        improved_rate_list4.append(rate4)

    print("unit01:", np.mean(improved_rate_list1))
    print("unit03:", np.mean(improved_rate_list2))
    print("unit11:", np.mean(improved_rate_list3))
    print("unit17:", np.mean(improved_rate_list4))
    x = range(1, 13)
    y1 = improved_rate_list1
    y2 = improved_rate_list2
    y3 = improved_rate_list3
    y4 = improved_rate_list4
    plt.plot(x, y1, marker='o', color='b')
    plt.plot(x, y2, marker='x', color='r')
    plt.plot(x, y3, marker='*', color='g')
    plt.plot(x, y4, marker='', color='orange')
    plt.show()


def get_chapter3_exe4_fig_rmse_mae(fed1,single1, fed2, single2, fed3, single3, fed4, single4):


    improved_rate_list1 = []
    improved_rate_list2 = []
    improved_rate_list3 = []
    improved_rate_list4 = []

    for i in range(len(fed1)):
        rate1 = (single1[i] - fed1[i]) / fed1[i]
        improved_rate_list1.append(rate1)

        rate2 = (single2[i] - fed2[i]) / fed2[i]
        improved_rate_list2.append(rate2)

        rate3 = (single3[i] - fed3[i]) / fed3[i]
        improved_rate_list3.append(rate3)

        rate4 = (single4[i] - fed4[i]) / fed4[i]
        improved_rate_list4.append(rate4)

    print("unit01:", np.mean(improved_rate_list1))
    print("unit03:", np.mean(improved_rate_list2))
    print("unit11:", np.mean(improved_rate_list3))
    print("unit17:", np.mean(improved_rate_list4))
    x = range(1, 13)
    y1 = improved_rate_list1
    y2 = improved_rate_list2
    y3 = improved_rate_list3
    y4 = improved_rate_list4
    plt.plot(x, y1, marker='o', color='b')
    plt.plot(x, y2, marker='x', color='r')
    plt.plot(x, y3, marker='*', color='g')
    plt.plot(x, y4, marker='', color='orange')
    plt.show()


def cal_fed_6basins_avg_error():
    model_name = "./MODEL-fed_hydro_6basins_t30_210-N(0).model"
    basin_ids = ['01030500', '01013500', '01031500', '01052500', '01054200', '01055000']
    nse_list = []
    rmse_list = []
    mae_list = []
    for basin_id in basin_ids:
        nse, rmse, mae = fed_model_test(model_name, basin_id)
        nse_list.append(nse)
        rmse_list.append(rmse)
        mae_list.append(mae)
    print("平均NSE", np.mean(nse_list))
    print("平均RMSE", np.mean(rmse_list))
    print("平均MAE", np.mean(mae_list))


def cal_local_6basins_avg_error():
    model_name = "./single_lstm_model/01/lstm_hydro_model_unit01_"
    basin_ids = ['01030500', '01013500', '01031500', '01052500', '01054200', '01055000']
    nse_list = []
    rmse_list = []
    mae_list = []
    for basin_id in basin_ids:
        model = model_name + str(basin_id) + ".model"
        nse, rmse, mae = single_model_test(model, basin_id)
        nse_list.append(nse)
        rmse_list.append(rmse)
        mae_list.append(mae)
    print("平均NSE", np.mean(nse_list))
    print("平均RMSE", np.mean(rmse_list))
    print("平均MAE", np.mean(mae_list))


def gen_4_units_box_fig_chapter_ex4():
    """
        第三章实验4， 箱型图
    """
    fed_nse1 = [0.8214513641286092, 0.7542472746270342, 0.8199971148188224, 0.7365483856720245, 0.601169267044835,
                0.6810456518023866, 0.7488897677700637, 0.7764051187190378, 0.7712855546506834, 0.7404529698241297,
                0.7519116349099357, 0.7379951455688617]

    fed_nse2 = [0.5517858170742502, 0.4308757697677188, 0.5433837343437413, 0.6430399349608251, 0.582808534925007,
                0.3224952947080141, 0.45641345128180233, 0.32158309621014471, 0.41607589548280166, 0.4381320366231674,
                0.4913633818737383, 0.5022140026161825]


    # unit 11
    fed_nse3 = [0.7628535051306087, 0.68537696704412, 0.5072069788268099, 0.6422739428784994, 0.660126426340937,
                0.758773592413981, 0.618880752228085, 0.7993294535467049, 0.3926644924743346, 0.5684630908271184,
                0.5069437188691708, 0.29396890459928837]


    # unit 17
    fed_nse4 = [0.5735658651436095, 0.8812296134524112, 0.8728627619027447, 0.49928377903446375, 0.8080258737686972,
                0.6987679045305333, 0.8942484993493829, 0.8185967778870663, 0.7505062082946831, 0.7702666137628462,
                0.5701215987835464, 0.6678072071943084]

    plt.figure(figsize=(11, 5.5))  # 设置画布的尺寸
    plt.rcParams.update({'font.family': 'Times New Roman', "font.size": 16})
    # plt.title('Examples of boxplot', fontsize=20)  # 标题，并设定字号大小
    labels = 'New England', 'South Atlantic-Gulf', 'Arkansas-White-Red', 'Pacific Northwest'  # 图例

    # vert=False 水平箱线图
    # showmeans=True 显示均值
    plt.boxplot([fed_nse1, fed_nse2, fed_nse3, fed_nse4], labels=labels, showmeans=True, showfliers=True, )
    plt.ylabel("NSE", fontdict={'family': 'Times New Roman', 'size': 16})
    plt.show()  # 显示图像


def chapter3_exe2_bar(num_list1, num_list2, num_list3, flag=1):

    label_list = ['01030500', '01013500', '01031500', '01052500', '01054200', '01055000']  # 横坐标刻度显示值
    # num_list1 = [0.43966531118442603, 0.7719611231758995, 0.7904479832157404, 0.7881718449795683]  # 纵坐标值1
    # num_list2 = [0.5281783317659351, 0.7736286558822844, 0.8035208903404784, 0.7985652511914886]  # 纵坐标值2
    # num_list3 = [0.6427281935196042, 0.769691437120537, 0.8091668794309137, 0.801814668841694]  # 纵坐标值3
    fed_model_5 = []
    fed_model_10 = []
    fed_model_20 = []
    x = range(len(num_list1))
    plt.figure(figsize=(12, 6), dpi=80)
    """
    绘制条形图
    left:长条形中点横坐标
    height:长条形高度
    width:长条形宽度，默认值0.8
    label:为后面设置legend准备
    """

    # color4 = ()
    width_bar = 0.15
    rects1 = plt.bar(x, height=num_list1, width=width_bar, alpha=0.8, color="darkgray", label="Local-LSTM")
    rects2 = plt.bar([i + width_bar + 0.05 for i in x], height=num_list2, width=width_bar, color="darkorange",
                     label="Fed-LSTM")
    rects3 = plt.bar([i + width_bar * 2 + 0.1 for i in x], height=num_list3, width=width_bar, color="#6488ea",
                     label="FedHydro")
    if flag == 1:
        plt.ylim(0, 1)  # y轴取值范围
    plt.xlabel('流域编号', fontdict={'family': 'SimSun', 'size': 18})
    if flag == 1:
        plt.ylabel('NSE', fontdict={'family': 'Times New Roman', 'size': 18})
    elif flag == 2:
        plt.ylabel('RMSE', fontdict={'family': 'Times New Roman', 'size': 18})
    else:
        plt.ylabel('MAE', fontdict={'family': 'Times New Roman', 'size': 18})
    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """
    plt.xticks([index + width_bar + 0.05 for index in x], label_list, FontProperties='Times New Roman', size=18)
    plt.yticks(FontProperties='Times New Roman', size=18)

    # plt.title("")
    # if flag == 1:
    #     plt.legend(prop={'family': 'Times New Roman', 'size': 15}, loc="upper left")  # 设置题注
    # else:
    #     plt.legend(prop={'family': 'Times New Roman', 'size': 15})  # 设置题注
    plt.legend(prop={'family': 'Times New Roman', 'size': 15})  # 设置题注
    plt.show()


if __name__ == '__main__':
    """第三章实验一"""
    # local_lstm = [0.766, 0.810, 0.769, 0.701, 0.549, 0.634]
    # fed_lstm = [0.588, 0.445, 0.707, 0.611, 0.454, 0.587]
    # fed_hydro = [0.774, 0.802, 0.815, 0.727, 0.583, 0.679]
    # chapter3_exe2_bar(local_lstm, fed_lstm, fed_hydro, flag=1)

    # local_lstm = [1.147, 0.893, 1.646, 2.063, 3.280, 2.349]
    # fed_lstm = [1.522, 1.527, 1.856, 2.354, 3.611, 2.495]
    # fed_hydro = [1.127, 0.895, 1.474, 1.972, 3.156, 2.201]
    # chapter3_exe2_bar(local_lstm, fed_lstm, fed_hydro, flag=2)
    # local_lstm = [0.633, 0.549, 0.854, 0.905, 1.414, 1.009]
    # fed_lstm = [0.909, 0.890, 0.924, 1.075, 1.642, 1.144]
    # fed_hydro = [0.612, 0.558, 0.742, 0.874, 1.304, 0.953]
    # chapter3_exe2_bar(local_lstm, fed_lstm, fed_hydro, flag=3)
    # cal_fed_6basins_avg_error()
    # cal_local_6basins_avg_error()
    """生成第三章实验结果"""
    # get_chapter3_exe4_fig()
    """rmse mae"""
    # rmse
    # fed1 = [1.002092392812496, 1.0158991472545589, 1.454205388515583, 1.936939973937449, 3.0861002571614002, 2.1932745529812303, 1.524708256905505, 1.6669742308015163, 1.2723241917173256, 1.7243352268017906, 1.4663925605902393, 1.3419907083775833]
    # single1 = [1.3017457845506175, 0.8972488176522565, 1.8342459594124576, 2.286713602365837, 3.145318539755057, 2.1463854770886166, 1.7939819456857622, 1.8916216023272965, 1.5359835696673447, 1.9494366029672037, 1.5872882158794257, 1.486205801845247]
    # fed2 = [1.5762970943481247, 1.9215675561446521, 1.1030159822163277, 0.883261391545017, 0.9857666851135845, 1.3538641425303257, 1.1378125546107996, 1.3409963067030566, 1.114997272238234, 1.2341032976848916, 0.9918924781963384, 1.0499431079087858]
    # single2 = [1.7853170632606792, 2.0176061508149092, 1.134402934512481, 1.0016129359323365, 1.0932675080833225, 1.2919815985969452, 1.268475465677406, 1.322100733183133, 1.0442836727389198, 1.0641759831235202, 1.0463554127071022, 1.1412352597529565]
    # fed3 = [1.904243086969125, 1.028206450556293, 2.8658521747586776, 1.3230244868107777, 0.9520797291063646, 0.855600288432978, 0.8662589328202671, 0.6980390131481226, 0.09088590974282808, 1.7207042289253358, 0.13128551443058936, 0.17526878154622744]
    #
    # single3 = [2.035461188504492, 1.1832418734974446, 2.9680789527456115, 1.2461010229361333, 1.039198710756004, 1.0367389398210762, 1.0233306497986736, 0.8886155940544209, 0.0668865797210185, 1.5292156448951995, 0.14039799866818853, 0.1573086665309262]
    # fed4 = [0.4887583220055174, 3.5977087133740757, 2.56116717081003, 8.186907959062433, 2.0721320597391264, 3.7661565605393386, 2.81738199000429, 5.470288974730744, 5.359714383650632, 5.835735770673793, 1.2944918828459386, 4.055234515069723]
    # single4 = [0.49132197218640067, 3.948074868943152, 2.762447419892521, 8.726773733309578, 2.20699955527642, 4.068631223005963, 2.808358987302642, 5.3323547771419895, 5.437200333359308, 6.072656496081797, 1.5564868125695506, 4.250496475684335]
    # fed1 = [0.5868416778427873, 0.6106631061184985, 0.7132160195331166, 0.8674361617461656, 1.2312042341670812, 0.9124216117650602, 0.7190042191272601, 0.7407880000298089, 0.6566668816284535, 0.8019863781114714, 0.6739264336177506, 0.6334294405053857]
    # single1 = [0.6942354926324971, 0.5403142240980632, 0.856484110729466, 0.9928832368120646, 1.3862812700291027, 0.9434559791238978, 0.8693129770068314, 0.7801870388692602, 0.6889627351924005, 0.8945600255943021, 0.7217720462931247, 0.6563275446565471]
    # fed2 = [0.5015755001445171, 0.5341429166440752, 0.4460180310765317, 0.43400042206610917, 0.5169397880844224, 0.6143648453621399, 0.5206596329319698, 0.561137872877111, 0.4731911847534192, 0.515684116748532, 0.6429030578392869, 0.5161699443368537]
    # single2 = [0.5456395264049467, 0.5400327042182111, 0.4412757835725686, 0.44171491739723917, 0.5506824943026097, 0.6138391578549149, 0.5718045505632947, 0.5369474140760678, 0.4406190751950495, 0.48214367206943937, 0.6510517335358504, 0.5174222780344352]
    # fed3 = [0.6287447900978453, 0.5104026199456412, 0.636484624702581, 0.48605736492988855, 0.5041611104415189, 0.389401989720558, 0.38718054043630046, 0.33001787860800563, 0.027800633590183442, 0.5111935393522071, 0.05649632942831955, 0.07016194842701115]
    #
    # single3 = [0.6180782609771196, 0.5586988886914981, 0.5796685349059767, 0.47700316599365267, 0.503503881351179, 0.40054679348093936, 0.3911465891694781, 0.42240037637225664, 0.02318926250967094, 0.4492281975312627, 0.058999209183717614, 0.06542109902358179]
    #
    # fed4 = [0.2589432938041159, 1.6168250859732207, 1.177091571264701, 1.7573693154032246, 0.9015888437894342, 1.5239795708297832, 1.393200587169362, 2.4436849007971193, 2.5013268302120863, 2.457999592044753, 0.7784006854069586, 2.173354264064607]
    # single4 = [0.2634135441604836, 1.7963404505933867, 1.2346698174391604, 1.8603338537641156, 0.8633857377022046, 1.6006375011944982, 1.2989859506994883, 2.3639559371057186, 2.5706283017866736, 2.5129513044617156, 0.9217969723416959, 2.2646687587719603]
    #
    # get_chapter3_exe4_fig_rmse_mae(fed1, single1, fed2, single2, fed3, single3, fed4, single4)


    """测试特征融合"""
    # fusion_feature_test("01013500")

    """验证单个模型的预测效果"""
    # model_path = "D:\\河海大学\\研究课题\\研究课题\\实验相关\\PSGD\\Parallel-SGD\\dataset\\" \
    #              "code_test\\hydro_lstm_test\\single_lstm_model\\01\\lstm_hydro_model_unit01_01013500.model"
    # baasin_id = "01013500"
    # get_trained_single_model_predict(model_path, baasin_id)
    # unit_code = "03"
    # basin_id = "02046000"
    # node_count = 12
    # epochs = [100]
    # for epoch in epochs:
    #     single_basin_train(unit_code, basin_id, epoch=epoch)

    """03 11 17"""
    # units = ["03", "11", "17"]
    # node_count = 12
    # epoch = 70
    # gen_unit_single_model(units, node_count, epoch)

    """01"""
    # units = ["01"]
    # node_count = 12
    # epoch = 70
    # gen_unit_single_model(units, node_count, epoch)

    # f = open('D:/河海大学/研究课题/水文预报/课题实验/Parallel-SGD/dataset/code_test
    # /test_result/6basins_210.csv', 'w', encoding='utf-8', newline="")
    # csv_writer = csv.writer(f)
    # csv_writer.writerow(["basin_id", "  ", "NSE", "RMSE", "MAE"])
    # csv_writer.writerow(["01030500", "  ", "", "", ""])
    # y = np.array([1, 2, 2, 3])
    # y_hat = np.array([1, 2, 2, 4])
    # MSE = np.mean(np.square(y - y_hat))
    # RMSE = np.sqrt(np.mean(np.square(y - y_hat)))
    # MAE = np.mean(np.abs(y - y_hat))
    # MAPE = np.mean(np.abs((y - y_hat) / y)) * 100
    # print(MSE)
    # print(RMSE)
    # print(MAE)
    # print(MAPE)
    # save_path = 'D:/河海大学/研究课题/水文预报/课题实验/Parallel-SGD/dataset/code_test/test_result/6basins_210.csv'
    # basin_ids = ['01030500', '01013500', '01031500', '01052500', '01054200', '01055000']
    "few shot retrain"
    # global_model_name = 'D:\河海大学\研究课题\研究课题\实验相关\PSGD\Parallel-SGD\dataset\code_test\hydro_lstm_test\MODEL-fed_hydro_6basins_t30_210-N(0).model'
    # # # save_csv(basin_ids, save_path, global_model_name)
    # # model_retrain(global_model_name, '01030500', 10)
    # nse, rmse, mae = model_few_shot_retrain(global_model_name, '01030500', 10)
    # print(nse)
    # print(rmse)
    # print(mae)
    # for basin_id in basin_ids:
    #     gen_retrain_model(global_model_name, basin_id, 10)

    """3流域，local_turn=1"""
    # in_basin_ids = ['01030500', '01013500', '01031500']
    # without_basin_ids = ['01134500', '01137500', '01139000', '01139800',
    # '01142500', '01144000', '01162500', '01169000', '01170100']
    # local_N1 = 2
    # get_3basins_local_1(in_basin_ids, without_basin_ids, local_N1)



    """对比实验3"""
    # global_fed_model = "D:\\河海大学\\研究课题\\研究课题\\实验相关\\PSGD\\Parallel-SGD\\dataset\\code_test\\" \
    #                    "hydro_lstm_test\\"
    # scene_num = 8
    # # in_basin_ids = ['01030500']
    # in_basin_ids = ['01030500', '01013500', '01031500']
    # local_N1 = 5
    # local_N2 = 10
    # local_N3 = 15
    #
    # # without_basin_ids = ['01134500']
    # # without_basin_ids = ['01134500', '01137500', '01139000', '01139800', '01142500', '01144000']
    # without_basin_ids = ['01134500', '01137500', '01139000', '01139800', '01142500', '01144000',
    #                     '01162500', '01169000', '01170100']
    # in_list_5, in_list_10, in_list_20, without_list_5,_without_list_10, without_list_20 = \
    #   gen_n_basins_fed_model_table(global_fed_model, scene_num, in_basin_ids, without_basin_ids, local_N1, local_N2, local_N3)
    # print(in_list_5, in_list_10, in_list_20, without_list_5,_without_list_10, without_list_20)

    """参与训练"""
    """nse:"""
    # 参与训练
    # num_list1 = [0.439665311, 0.771961123, 0.790447983,  0.788171845]  # 纵坐标值1
    # num_list2 = [0.528178332, 0.773628656, 0.80352089,  0.798565251]  # 纵坐标值2
    # num_list3 = [0.598460273, 0.774111574, 0.803496438, 0.794998889]  # 纵坐标值3
    # gen_n_basins_fed_model_bar_chart(num_list1, num_list2, num_list3, flag=1)
    # # #
    # # # """rmse"""
    # # # # 参与训练
    # num_list1 = [1.97751111, 1.221323073, 1.186328862, 1.183604901]  # 纵坐标值1
    # num_list2 = [1.817564321, 1.22078922, 1.154791178, 1.157398976]  # 纵坐标值2
    # num_list3 = [1.678122, 1.22195776, 1.161529344, 1.175969431]  # 纵坐标值3
    # gen_n_basins_fed_model_bar_chart(num_list1, num_list2, num_list3, flag=2)
    # # #
    # # # """mae"""
    # # # # 参与训练
    # num_list1 = [1.080998677, 0.691360342, 0.670741618, 0.655902936]  # 纵坐标值1
    # num_list2 = [0.970957409, 0.680717484, 0.658398602, 0.636906934]  # 纵坐标值2
    # num_list3 = [0.895772622, 0.672190168, 0.656825695, 0.64615149]  # 纵坐标值3
    # gen_n_basins_fed_model_bar_chart(num_list1, num_list2, num_list3, flag=3)

    """未参与训练"""
    # "nse"
    # # # 未参与训练
    # num_list1 = [0.314199028, 0.631351429, 0.647194684, 0.694655654]  # 纵坐标值1
    # num_list2 = [0.370913399, 0.628777492, 0.642059131, 0.693768348]  # 纵坐标值2
    # num_list3 = [0.438295755, 0.623229249, 0.634414548, 0.689756322]  # 纵坐标值3
    # gen_n_basins_fed_model_bar_chart(num_list1, num_list2, num_list3, flag=1)
    # #
    # # "rmse"
    # # # # 未参与训练
    # num_list1 = [2.183913137, 1.584063216, 1.548923656, 1.447466501]  # 纵坐标值1
    # num_list2 = [2.086744394, 1.591603519, 1.554582167, 1.44677523]  # 纵坐标值2
    # num_list3 = [1.968525426, 1.605212144, 1.569802342, 1.456379524]  # 纵坐标值3
    # gen_n_basins_fed_model_bar_chart(num_list1, num_list2, num_list3, flag=2)
    # #
    # # "mae"
    # num_list1 = [1.118715565, 0.791435246, 0.791016089, 0.747978897]  # 纵坐标值1
    # num_list2 = [1.056863848, 0.789160549, 0.792984764, 0.743590192]  # 纵坐标值2
    # num_list3 = [0.981256707, 0.792681056, 0.792554521, 0.744474788]  # 纵坐标值3
    # gen_n_basins_fed_model_bar_chart(num_list1, num_list2, num_list3, flag=3)


    # global_model_name = 'D:\\河海大学\\研究课题\\研究课题\\实验相关\\PSGD\\Parallel-SGD\\dataset\\code_test\\' \
    #                     'hydro_lstm_test\\MODEL-fed_hydro_6basins_t30_210-N(0).model'
    # model_retrain(global_model_name, '01013500', 5)


    """"第三章 实验2"""
    # 运行产生2个流域的预测结果
    # basin_ids = ['01030500', '01013500']
    # basin_ids = ['01031500', '01052500']
    # basin_ids = ['01054200', '01055000']
    # gen_fed_hydro_two_basin_pred(basin_ids)

    """生成第三章实验四的实验结果"""
    unit_codes = ["01"]
    node_count = 12
    #
    single_model_path = "./single_lstm_model"
    for unit_code in unit_codes:
        fed_model_name = "./fed_model/MODEL-fed_hydro_unit" + str(unit_code) + "_12basins_t30_210-N(11).model"
        gen_4units_model_result(unit_code, node_count, fed_model_name, single_model_path)

    """第三章 实验4 箱型图"""
    # gen_4_units_box_fig_chapter_ex4()



