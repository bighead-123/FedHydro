import nn
from utils.constants import model_path, model_drop_path
from dataset.code_test.utils_test.get_hydro_data import GetHydroData
from nn.utils.nse_util import calc_nse
from nn.utils.rmse_util import cal_rmse
from nn.utils.mae_util import cal_mae

hidden_size = 20  # Number of LSTM cells
dropout_rate = 0  # Dropout rate of the final fully connected Layer [0.0, 1.0]
learning_rate = 0.001  # Learning rate used to update the weights
sequence_length = 30  # Length of the meteorological record provided to the network
batch_size = 256


# 单个模型基于本地数据训练
def single_basin_train(unit_code, basin_id, epoch, single_model):
    getHydroData = GetHydroData(basin_id, sequence_length)
    train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()
    # single_model = nn.model.SequentialModel.load(single_model_path)
    single_model.fit(train_x, train_y, epoch=epoch, batch_size=256)
    # single_model.fit_early_stopping(train_x, train_y, val_x, val_y, exp_epoch=50, req_stop_num=30, epoch=epoch, batch_size=256)
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


def model_test(model, test_x, test_y, ds_test):
    predict_y = model.predict(test_x)
    predict_y = ds_test.local_rescale(predict_y, variable='output')
    # 计算NSE
    nse = calc_nse(test_y, predict_y)
    # 计算RMSE
    rmse = cal_rmse(test_y, predict_y)
    mae = cal_mae(test_y, predict_y)

    return nse, rmse, mae


def get_hydro_data(basin_id):
    getHydroData = GetHydroData(basin_id, sequence_length)
    train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()
    ds_test = getHydroData.get_ds_test()
    train_data = []
    train_data.append(train_x)
    train_data.append(train_y)

    test_data = []
    test_data.append(test_x)
    test_data.append(test_y)

    return train_data, test_data, ds_test


def build_model():
    model = nn.model.Model.load(model_path)
    return model


def tl_a_multi_basins(all_train_data, model, epoch, batch_size):
    """联合训练"""
    train_x = all_train_data[0]
    train_y = all_train_data[1]
    model.fit(train_x, train_y, epoch, batch_size)
    global_model = model
    return global_model


def tl_a_fune(global_model, single_train_data, epoch, batch_size):
    # lstm_weights(26, 80)---(1+5+20, 4*20)
    # (6:26, 80) 为w_ih,w_gh,w_fh,w_oh
    train_x = single_train_data[0]
    train_y = single_train_data[1]
    # test_x = single_test_data[0]
    # test_y = single_test_data[1]
    # nse, rmse, mae = model_test(global_model, test_x, test_y, ds_test)
    # print("微调前：nse:", nse)
    for i in range(epoch):
        weights_list = global_model.get_weights_value()
        global_model.fit(train_x, train_y, epoch=1, batch_size=batch_size)
        global_model.update_parameter_fune(weights_list, fune_patten=1)
        weights_list1 = global_model.get_weights_value()
    # nse, rmse, mae = model_test(global_model, test_x, test_y, ds_test)
    model = global_model
    return model


def tl_b_fune(global_model, single_train_data, epoch, batch_size):
    """更新LSTM内部wh，以及输出"""
    # lstm_weights(26, 80)---(1+5+20, 4*20)
    # (6:26, 80) 为w_ih,w_gh,w_fh,w_oh
    # weights_list:lstm(), w,b
    train_x = single_train_data[0]
    train_y = single_train_data[1]
    # test_x = single_test_data[0]
    # test_y = single_test_data[1]
    # nse, rmse, mae = model_test(global_model, test_x, test_y, ds_test)
    # print("微调前：nse:", nse)
    for i in range(epoch):
        weights_list = global_model.get_weights_value()
        global_model.fit(train_x, train_y, epoch=1, batch_size=batch_size)
        global_model.update_parameter_fune(weights_list, fune_patten=2)
        weights_list1 = global_model.get_weights_value()
    # nse, rmse, mae = model_test(global_model, test_x, test_y, ds_test)
    # print("微调后：nse:", nse)
    model = global_model
    return model


def tl_c_fune(global_model, single_train_data, epoch, batch_size, single_test_data, ds_test):
    """所有参数都进行更新"""
    # lstm_weights(26, 80)---(1+5+20, 4*20)
    # (6:26, 80) 为w_ih,w_gh,w_fh,w_oh
    train_x = single_train_data[0]
    train_y = single_train_data[1]
    test_x = single_test_data[0]
    test_y = single_test_data[1]
    nse, rmse, mae = model_test(global_model, test_x, test_y, ds_test)
    print("微调前：nse:", nse)
    global_model.fit(train_x, train_y, epoch=epoch, batch_size=batch_size)
    nse, rmse, mae = model_test(global_model, test_x, test_y, ds_test)
    print("微调后：nse:", nse)


def tl_a_test():
    model = nn.model.Model.load('../model/MODEL-fed_hydro_6basins-N(0).model')
    model.compile(nn.gradient_descent.ADAMOptimizer(alpha=learning_rate))
    basin_id = "01030500"
    epoch = 10
    train_data, test_data, ds_test =get_hydro_data(basin_id)
    tl_a_fune(model, train_data, epoch, batch_size, test_data, ds_test)


def tl_b_test():
    model = nn.model.Model.load('../model/MODEL-fed_hydro_6basins-N(0).model')
    model.compile(nn.gradient_descent.ADAMOptimizer(alpha=learning_rate))
    basin_id = "01030500"
    epoch = 10
    train_data, test_data, ds_test =get_hydro_data(basin_id)
    tl_b_fune(model, train_data, epoch, batch_size, test_data, ds_test)


def tl_c_test():
    model = nn.model.Model.load('../model/MODEL-fed_hydro_6basins-N(0).model')
    model.compile(nn.gradient_descent.ADAMOptimizer(alpha=learning_rate))
    basin_id = "01030500"
    epoch = 10
    train_data, test_data, ds_test = get_hydro_data(basin_id)
    tl_c_fune(model, train_data, epoch, batch_size, test_data, ds_test)


if __name__ == '__main__':
    # model = nn.model.Model.load(model_drop_path)
    # epochs = [40, 50, 60, 70]
    # for epoch in epochs:
    #     single_basin_train("01", "01030500", epoch, model)
    # print(model.summary())
    # tl_a_test()
    # tl_b_test()
    tl_c_test()