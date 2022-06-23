from dataset.code_test.utils_test.get_hydro_data import GetHydroData
import nn
from nn.utils.nse_util import calc_nse
from nn.utils.rmse_util import cal_rmse
from nn.utils.mae_util import cal_mae
import csv
import matplotlib.pyplot as plt
import numpy as np

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


# 模型再测试记录
def gen_retrain_model(global_model_name, basin_id, epoch):
    path = 'D:/河海大学/研究课题/水文预报/课题实验/Parallel-SGD/dataset/code_test/hydro_lstm_test/fed_model_retrain/'
    getHydroData = GetHydroData(basin_id, sequence_length)
    train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()
    model = nn.model.Model.load(global_model_name)
    model.compile(nn.gradient_descent.ADAMOptimizer(alpha=learning_rate))
    # 微调全局模型
    model.fit(train_x, train_y, epoch=epoch, batch_size=batch_size)
    model.save(str(path)+'MODEL-fed_hydro_6basins-N(0)_'+str(basin_id)+'_retrain10.model')


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


def gen_fig_res(basin_ids):
    path = 'D:/河海大学/研究课题/水文预报/课题实验/Parallel-SGD/dataset/code_test/hydro_lstm_test/'
    i = 1
    fig, axes = plt.subplot(3, 2)  # 2行1列，第一张图
    for basin_id in basin_ids:
        getHydroData = GetHydroData(basin_id, sequence_length)
        train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()
        single_lstm_model = nn.model.Model.load(path+'single_lstm_model/lstm_hydro_'+str(basin_id)+'.model')
        fed_lstm_model = nn.model.Model.load(path+'fed_model/MODEL-fed_hydro_6basins_t30_210-N(0).model')
        fed_lstm_model_10 = nn.model.Model.load(path+'fed_model_retrain/MODEL-fed_hydro_6basins-N(0)_'+str(basin_id)+'_retrain10.model')
        ds_test = getHydroData.get_ds_test()
        # lstm model
        single_predict_y = single_lstm_model.predict(test_x)
        single_predict_y = ds_test.local_rescale(single_predict_y, variable='output')
        single_nse = calc_nse(test_y, single_predict_y)  # 计算NSE
        # test_y = test_y[:1000]
        # single_predict_y = single_predict_y[:1000]
        # test_y = ds_test.reshape_discharge(test_y)
        single_predict_y = ds_test.reshape_discharge(single_predict_y)

        # fed_lstm
        fed_predict_y = fed_lstm_model.predict(test_x)
        fed_predict_y = ds_test.local_rescale(fed_predict_y, variable='output')
        fed_nse = calc_nse(test_y, fed_predict_y)  # 计算NSE
        # test_y = test_y[:1000]
        # fed_predict_y = fed_predict_y[:1000]
        # test_y = ds_test.reshape_discharge(test_y)
        fed_predict_y = ds_test.reshape_discharge(fed_predict_y)

        # fed-lstm-10
        fed10_predict_y = fed_lstm_model_10.predict(test_x)
        fed10_predict_y = ds_test.local_rescale(fed10_predict_y, variable='output')
        fed10_nse = calc_nse(test_y, fed10_predict_y)  # 计算NSE
        # test_y = test_y[:1000]
        # fed10_predict_y = fed10_predict_y[:1000]
        # test_y = ds_test.reshape_discharge(test_y)
        fed10_predict_y = ds_test.reshape_discharge(fed10_predict_y)

        print("model-worker0,NSE:", single_nse)
        print("model-worker0,fed_NSE:", fed_nse)
        print("model-worker0,fed10_NSE:", fed10_nse)

        axes[0, i].plot(list(range(len(test_y))), test_y, color='b', label='Observed')
        axes[0, i].plot(list(range(len(single_predict_y))), single_predict_y, color='orange', label='LSTM Model')
        axes[0, i].plot(list(range(len(fed_predict_y))), fed_predict_y, color='green', label='Fed-LSTM')
        axes[0, i].plot(list(range(len(fed10_predict_y))), fed10_predict_y, color='red', label='Fed-LSTM-10')
        # plt.title(f"basin:{basin_id}, NSE:{nse:.3f}")
        axes[0, i].xlabel('day number')
        axes[0, i].ylabel('discharge')
        axes[0, i].legend()
        axes[0, i].savefig(path+'fig_res/' + str(basin_id) + '.svg', format='svg')
        i += 1
    plt.show()


def gen_single_basin_pred(basin_id):
    config = {
        "font.family": 'serif',
        # "font.size": 12,
        "mathtext.fontset": 'stix',
        # "font.serif": ['SimSun'],
    }
    plt.rcParams.update(config)

    path = 'D:/河海大学/研究课题/水文预报/课题实验/Parallel-SGD/dataset/code_test/hydro_lstm_test/'
    getHydroData = GetHydroData(basin_id, sequence_length)
    train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()
    single_lstm_model = nn.model.Model.load(path + 'single_lstm_model/lstm_hydro_' + str(basin_id) + '.model')
    fed_lstm_model = nn.model.Model.load(path + 'fed_model/MODEL-fed_hydro_6basins_t30_210-N(0).model')
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
    single_predict_y = ds_test.reshape_discharge(single_predict_y)

    # fed_lstm
    fed_predict_y = fed_lstm_model.predict(test_x)
    fed_predict_y = ds_test.local_rescale(fed_predict_y, variable='output')
    fed_nse = calc_nse(test_y, fed_predict_y)  # 计算NSE
    # test_y = test_y[:1000]
    # fed_predict_y = fed_predict_y[:1000]
    # test_y = ds_test.reshape_discharge(test_y)
    fed_predict_y = ds_test.reshape_discharge(fed_predict_y)

    # fed-lstm-10
    fed10_predict_y = fed_lstm_model_10.predict(test_x)
    fed10_predict_y = ds_test.local_rescale(fed10_predict_y, variable='output')
    fed10_nse = calc_nse(test_y, fed10_predict_y)  # 计算NSE
    # test_y = test_y[:1000]
    # fed10_predict_y = fed10_predict_y[:1000]
    # test_y = ds_test.reshape_discharge(test_y)
    fed10_predict_y = ds_test.reshape_discharge(fed10_predict_y)

    test_y = ds_test.reshape_discharge(test_y)
    plt.rcParams['figure.figsize'] = (8.0, 6.0)
    # plt.rcParams['figure.dpi'] = 120
    # plt.rcParams['savefig.dpi'] = 120
    plt.subplot(3, 1, 1)  # 2行1列，第一张图
    plt.plot(list(range(len(single_predict_y))), single_predict_y, color='orange', label='LSTM-Model')
    plt.plot(list(range(len(test_y))), test_y, color='b', label='Actual')
    # plt.title(f"basin:{basin_id}, NSE:{single_nse:.3f}")
    # plt.xlabel('day number')
    plt.ylabel('discharge')
    plt.legend(frameon=True, loc="upper left", fontsize='x-small')
    # plt.savefig(path+'/fig_res/' + str(basin_id) + '_lstm.svg', format='svg')

    plt.subplot(3, 1, 2)  # 2行1列，第一张图
    plt.plot(list(range(len(fed_predict_y))), fed_predict_y, color='green', label='Fed-LSTM')
    plt.plot(list(range(len(test_y))), test_y, color='b', label='Actual')
    # plt.title(f"basin:{basin_id}, NSE:{fed_nse:.3f}")
    # plt.xlabel('day number')
    plt.ylabel('discharge')
    plt.legend(frameon=True, loc="upper left", fontsize='x-small')
    # plt.savefig(path+'/fig_res/' + str(basin_id) + '_fed_lstm.svg', format='svg')

    plt.subplot(3, 1, 3)  # 2行1列，第一张图
    plt.plot(list(range(len(fed10_predict_y))), fed10_predict_y, color='r', label='Fed-LSTM-10')
    plt.plot(list(range(len(test_y))), test_y, color='b', label='ACtual')
    # plt.title(f"basin:{basin_id}, NSE:{fed10_nse:.3f}")
    plt.xlabel('day number')
    plt.ylabel('discharge')
    plt.legend(frameon=True, loc="upper left", fontsize='x-small')
    plt.savefig(path+'/fig_res/' + str(basin_id) + '_model_compare.svg', format='svg')
    plt.show()


if __name__ == '__main__':

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
    save_path = 'D:/河海大学/研究课题/水文预报/课题实验/Parallel-SGD/dataset/code_test/test_result/6basins_210.csv'
    basin_ids = ['01030500', '01013500', '01031500', '01052500', '01054200', '01055000']
    global_model_name = 'D:/河海大学/研究课题/水文预报/课题实验/Parallel-SGD/dataset/code_test/MODEL-fed_hydro_6basins_t30_210-N(0).model'
    # save_csv(basin_ids, save_path, global_model_name)
    # gen_retrain_model(global_model_name, '01030500', 1)
    # for basin_id in basin_ids:
    #     gen_retrain_model(global_model_name, basin_id, 10)
    # basin_ids = ['01030500']
    # gen_fig_res(basin_ids)
    gen_single_basin_pred('01030500')

