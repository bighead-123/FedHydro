from dataset import HydroDataSet
from nn.utils.nse_util import calc_nse
from nn.utils.rmse_util import cal_rmse
from nn.utils.mae_util import cal_mae


def model_test(model, test_x, test_y, ds_test):
    predict_y = model.predict(test_x)
    predict_y = ds_test.local_rescale(predict_y, variable='output')
    # 计算NSE
    nse = calc_nse(test_y, predict_y)
    # 计算RMSE
    rmse = cal_rmse(test_y, predict_y)
    mae = cal_mae(test_y, predict_y)

    return nse, rmse, mae


def train_mutil_fune(global_model, single_train_data, epoch, batch_size):
    """所有参数都进行更新"""
    # lstm_weights(26, 80)---(1+5+20, 4*20)
    # (6:26, 80) 为w_ih,w_gh,w_fh,w_oh
    train_x = single_train_data[0]
    train_y = single_train_data[1]
    global_model.fit(train_x, train_y, epoch=epoch, batch_size=batch_size)
    return global_model