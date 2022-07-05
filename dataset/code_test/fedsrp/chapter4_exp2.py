from dataset.code_test.fed_fomo.local_lstm.local_lstm import train_local_lstm
from dataset.code_test.fed_meta_ref.MAML.data_util import HydroDataMeta
from dataset.code_test.fed_fomo.chapter_exp1 import get_bayesianlstm, get_limited_data
from dataset.code_test.fed_fomo.mutil_fune.mutil_fune import train_mutil_fune
from dataset.code_test.fed_fomo.fedhydro.fedhydro import train_fedhydro
from dataset.code_test.fed_fomo.transfer.transfer import tl_a_fune, tl_b_fune
from dataset.code_test.hydro_lstm_test.save_script_ import cal_nse_rmse_mae
from dataset.code_test.fed_fomo.fed_fomo import FedFomo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nn
from utils.constants import local_basic_lstm


local_basic_model_name = local_basic_lstm
# local_model = get_bayesianlstm()
model_name = './model/basic_local_model.model'
learning_rate = 0.001
# local_model.save(model_name)
# local_model = nn.model.SequentialModel.load(model_name)
# global_model = None
sequence_length = 30
# basin_id = "01052500"
date_range = {
        'train_date': {
            'start_date': pd.to_datetime("1980-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("1983-09-30", format="%Y-%m-%d")  # 三年训练期
        },
        'val_date': {
            'start_date': pd.to_datetime("2001-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2003-09-30", format="%Y-%m-%d")  # 测试日期1，差异性大， 数据太少不设置验证期
        },
        'test_date': {
            'start_date': pd.to_datetime("2003-10-01", format="%Y-%m-%d"),  # 测试日期2， 差异性小
            'end_date': pd.to_datetime("2005-09-30", format="%Y-%m-%d")
        },
    }

# 01047000
date_range1 = {
        'train_date': {
            'start_date': pd.to_datetime("1980-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("1982-09-30", format="%Y-%m-%d")  # 三年训练期
        },
        'val_date': {
            'start_date': pd.to_datetime("2008-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2010-09-30", format="%Y-%m-%d")  # 测试日期1，差异性大， 数据太少不设置验证期
        },
        'test_date': {
            'start_date': pd.to_datetime("2006-10-01", format="%Y-%m-%d"),  # 测试日期2， 差异性小
            'end_date': pd.to_datetime("2008-09-30", format="%Y-%m-%d")
        },
    }

# 01054200
date_range2 = {
        'train_date': {
            'start_date': pd.to_datetime("1980-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("1982-09-30", format="%Y-%m-%d")  # 三年训练期
        },
        'val_date': {
            'start_date': pd.to_datetime("2002-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2004-09-30", format="%Y-%m-%d")  # 测试日期1，差异性大， 数据太少不设置验证期
        },
        'test_date': {
            'start_date': pd.to_datetime("2000-10-01", format="%Y-%m-%d"),  # 测试日期2， 差异性小
            'end_date': pd.to_datetime("2002-09-30", format="%Y-%m-%d")
        },
    }
# 01055000
date_range3 = {
        'train_date': {
            'start_date': pd.to_datetime("1980-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("1982-09-30", format="%Y-%m-%d")  # 三年训练期
        },
        'val_date': {
            'start_date': pd.to_datetime("2002-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2004-09-30", format="%Y-%m-%d")  # 测试日期1，差异性大， 数据太少不设置验证期
        },
        'test_date': {
            'start_date': pd.to_datetime("2004-10-01", format="%Y-%m-%d"),  # 测试日期2， 差异性小
            'end_date': pd.to_datetime("2006-09-30", format="%Y-%m-%d")
        },
    }


rich_date_range = {
        'train_date': {
            'start_date': pd.to_datetime("1980-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("1995-09-30", format="%Y-%m-%d")  # 三年训练期
        },
        'val_date': {
            'start_date': pd.to_datetime("2000-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2005-09-30", format="%Y-%m-%d")  # 测试日期1，差异性大， 数据太少不设置验证期
        },
        'test_date': {
            'start_date': pd.to_datetime("2006-10-01", format="%Y-%m-%d"),  # 测试日期2， 差异性小
            'end_date': pd.to_datetime("2008-09-30", format="%Y-%m-%d")
        },
    }
# train_x, train_y, val_x, val_y, ds_val, test_x, test_y, ds_test = get_limited_data(basin_id, date_range)
# epoch = 100
# batch_size = 64


def train_global_model_with_sparse(global_epoch, global_batch_size, block_ids, sparse_train_x, sparse_train_y):
    data_path = "D:\\河海大学\\研究课题\\研究课题\\实验相关\\PSGD\\Parallel-SGD\\dataset\\code_test\\fed_fomo\\data"
    unit_code = "01"
    hydro_data = HydroDataMeta(data_path)
    node_count = len(block_ids)
    model = nn.model.SequentialModel.load(model_name)
    y_range = np.arange(1, global_epoch+1)
    all_train_x, all_train_y, all_test_x, all_test_y = hydro_data.load_data(unit_code, node_count)
    all_train_x = np.concatenate((all_train_x, sparse_train_x), axis=0)
    all_train_y = np.concatenate((all_train_y, sparse_train_y), axis=0)
    epoch_val_loss_list = []
    for i in range(1, global_epoch+1):
        model.fit(all_train_x, all_train_y, 1, global_batch_size)
        epoch_val_loss = model.evaluate_sum_loss(all_test_x, all_test_y, global_batch_size)
        epoch_val_loss = epoch_val_loss / all_test_x.shape[0]
        epoch_val_loss_list.append(epoch_val_loss)
    plt.plot(y_range, epoch_val_loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    model.save('./model/global_model_'+str(node_count)+'basins_'+str(global_epoch)+'epoch_2years.model')


# def train_global_model(global_epoch, global_batch_size, block_ids, sparse_train_x, sparse_train_y):
#     data_path = "D:\\河海大学\\研究课题\\研究课题\\实验相关\\PSGD\\Parallel-SGD\\dataset\\code_test\\fed_fomo\\data"
#     unit_code = "01"
#     hydro_data = HydroDataMeta(data_path)
#     node_count = len(block_ids)
#     model = nn.model.SequentialModel.load(model_name)
#     y_range = np.arange(1, global_epoch+1)
#     all_train_x, all_train_y, all_test_x, all_test_y = hydro_data.load_data(unit_code, node_count)
#     epoch_val_loss_list = []
#     for i in range(1, global_epoch+1):
#         model.fit(all_train_x, all_train_y, 1, global_batch_size)
#         epoch_val_loss = model.evaluate_sum_loss(all_test_x, all_test_y, global_batch_size)
#         epoch_val_loss = epoch_val_loss / all_test_x.shape[0]
#         epoch_val_loss_list.append(epoch_val_loss)
#     plt.plot(y_range, epoch_val_loss_list)
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.show()
#
#     # predict_y = model.predict(all_test_x)
#     # predict_y = ds_test.local_rescale(predict_y, variable='output')
#     # nse, rmse, mae = cal_nse_rmse_mae(test_y, predict_y)
#     # print(nse)
#     model.save('./model/global_model_'+str(node_count)+'basins_'+str(global_epoch)+'epoch.model')


def global_model_test(basin_id, global_model_name):

    date_range = {
        'train_date': {
            'start_date': pd.to_datetime("1980-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("1985-09-30", format="%Y-%m-%d")
        },
        'val_date': {
            'start_date': pd.to_datetime("1986-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("1987-09-30", format="%Y-%m-%d")
        },
        'test_date': {
            'start_date': pd.to_datetime("2003-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2005-09-30", format="%Y-%m-%d")  # 01-03相对预测效果要差，属于与训练期间水文模型相差较大的年份
        },
    }
    train_x, train_y, val_x, val_y, ds_val, test_x, test_y, ds_test =get_sparse_data(basin_id, date_range)
    global_model = nn.model.SequentialModel.load(global_model_name)
    predict_y = global_model.predict(test_x)
    predict_y = ds_test.local_rescale(predict_y, variable='output')
    nse, rmse, mae = cal_nse_rmse_mae(test_y, predict_y)
    print("nse:", nse)


def get_sparse_data(basin_id, date_range):
    train_x, train_y, val_x, val_y, ds_val, test_x, test_y, ds_test = get_limited_data(basin_id, date_range)
    return train_x, train_y, val_x, val_y, ds_val, test_x, test_y, ds_test


def fed_fomo(single_basin_data):
    pass


def local_lstm(basin_id, local_basic_model_name, single_basin_data, test_x, test_y, single_ds_test, epoch,  batch_size):
    local_basic_model = nn.model.SequentialModel.load(local_basic_model_name)
    model = train_local_lstm(local_basic_model, single_basin_data, epoch, batch_size)
    predict_y = model.predict(test_x)
    predict_y = single_ds_test.local_rescale(predict_y, variable='output')
    nse, rmse, mae = cal_nse_rmse_mae(test_y, predict_y)
    print("basin:", basin_id, ",nse:", nse)
    # model.save('./model/local_lstm_'+str(basin_id)+'_2years.model')
    # model.save('./model/local_lstm_'+str(basin_id)+'.model')


def local_lstm_test(basin_id, local_model_name, local_data_range):
    train_x, train_y, val_x, val_y, ds_val, test_x, test_y, ds_test = get_sparse_data(basin_id, local_data_range)
    loacl_lstm = nn.model.SequentialModel.load(local_model_name)
    predict_y = loacl_lstm.predict(test_x)
    predict_y = ds_test.local_rescale(predict_y, variable='output')
    nse, rmse, mae = cal_nse_rmse_mae(test_y, predict_y)
    print("nse:", nse)


def mutil_fune(global_model_name, basin_id, single_basin_data, epoch, batch_size, the_data_range):
    train_x, train_y, val_x, val_y, ds_val, test_x, test_y, ds_test = get_sparse_data(basin_id, the_data_range)
    global_model = nn.model.SequentialModel.load(global_model_name)
    predict_y = global_model.predict(val_x)
    predict_y = ds_test.local_rescale(predict_y, variable='output')
    nse, rmse, mae = cal_nse_rmse_mae(val_y, predict_y)
    print("mutil-fune, before fune:", nse)
    mutil_fune_model = train_mutil_fune(global_model, single_basin_data, epoch, batch_size)
    predict_y = mutil_fune_model.predict(val_x)
    predict_y = ds_test.local_rescale(predict_y, variable='output')
    nse, rmse, mae = cal_nse_rmse_mae(val_y, predict_y)
    print("mutil-fune, after fune:", nse)
    mutil_fune_model.save('./model/compare_model_0413/mutil_fune'+str(basin_id)+'_2years.model')
    # mutil_fune_model.save('./model/mutil_fune'+str(basin_id)+'_2years2.model')
    # mutil_fune_model.save('./model/mutil_fune'+str(basin_id)+'_2years.model')
    # mutil_fune_model.save('./model/mutil_fune'+str(basin_id)+'.model')
    return mutil_fune_model


def fed_hydro(fed_model_name, single_basin_data, epoch, batch_size, basin, test_x, test_y, ds_test):
    fed_model = nn.model.SequentialModel.load(fed_model_name)
    fed_model.compile(nn.gradient_descent.ADAMOptimizer(alpha=learning_rate))
    predict_y = fed_model.predict(test_x)
    predict_y = ds_test.local_rescale(predict_y, variable='output')
    nse, rmse, mae = cal_nse_rmse_mae(test_y, predict_y)
    print("fed_hydro, before fune:", nse)
    fed_hydro_model = train_fedhydro(fed_model, single_basin_data, epoch, batch_size)
    predict_y = fed_hydro_model.predict(test_x)
    predict_y = ds_test.local_rescale(predict_y, variable='output')
    nse, rmse, mae = cal_nse_rmse_mae(test_y, predict_y)
    print("fed_hydro, after fune:", nse)
    fed_hydro_model.save('./model/compare_model_0413/fed_hydro'+str(basin)+'.model')
    # fed_hydro_model.save('./model/fed_hydro'+str(basin)+'.model')
    return fed_hydro_model


def transfer_a(global_model_name, basin, single_basin_data, epoch, batch_size, test_x, test_y, ds_test):
    tl_model = nn.model.SequentialModel.load(global_model_name)
    predict_y = tl_model.predict(test_x)
    predict_y = ds_test.local_rescale(predict_y, variable='output')
    nse, rmse, mae = cal_nse_rmse_mae(test_y, predict_y)
    print("transfer-a, before fune:", nse)
    tl_a_model = tl_a_fune(tl_model, single_basin_data, epoch, batch_size)
    predict_y = tl_a_model.predict(test_x)
    predict_y = ds_test.local_rescale(predict_y, variable='output')
    nse, rmse, mae = cal_nse_rmse_mae(test_y, predict_y)
    print("transfer-a, after fune:", nse)
    tl_a_model.save('./model/compare_model_0413/tl_a_basin'+str(basin)+'_2years.model')
    # tl_a_model.save('./model/tl_a_basin'+str(basin)+'_2years2.model')
    # tl_a_model.save('./model/tl_a_basin'+str(basin)+'_2years.model')
    # tl_a_model.save('./model/tl_a_basin'+str(basin)+'.model')
    return tl_a_model


def transfer_b(global_model_name, basin, single_basin_data, epoch, batch_size, test_x, test_y, ds_test):
    tl_model = nn.model.SequentialModel.load(global_model_name)
    predict_y = tl_model.predict(test_x)
    predict_y = ds_test.local_rescale(predict_y, variable='output')
    nse, rmse, mae = cal_nse_rmse_mae(test_y, predict_y)
    print("transfer-b, before fune:", nse)
    tl_b_model = tl_b_fune(tl_model, single_basin_data, epoch, batch_size)
    predict_y = tl_b_model.predict(test_x)
    predict_y = ds_test.local_rescale(predict_y, variable='output')
    nse, rmse, mae = cal_nse_rmse_mae(test_y, predict_y)
    print("transfer-b, after fune:", nse)
    tl_b_model.save('./model/compare_model_0413/tl_b_basin' + str(basin) + '_2years.model')
    # tl_b_model.save('./model/tl_b_basin' + str(basin) + '_2years2.model')
    # tl_b_model.save('./model/tl_b_basin' + str(basin) + '_2years.model')
    # tl_b_model.save('./model/tl_b_basin' + str(basin) + '.model')
    return tl_b_model


def fed_srp_local_compare(basin_ids, sparse_date_range):
    nse1_list = []
    rmse1_list = []
    mae1_list = []

    nse2_list = []
    rmse2_list = []
    mae2_list = []
    for basin in basin_ids:
        train_x, train_y, val_x, val_y, ds_val, test_x, test_y, ds_test = get_sparse_data(basin, sparse_date_range)
        local_lstm_path = './model/local_lstm_' + str(basin) + '.model'
        fed_fomo_path = './model/fedfomo_8basins_client7.model'
        local_lstm_model = nn.model.SequentialModel.load(local_lstm_path)
        predict_y1 = local_lstm_model.predict(val_x)
        predict_y1 = ds_test.local_rescale(predict_y1, variable='output')
        nse1, rmse1, mae1 = cal_nse_rmse_mae(val_y, predict_y1)
        nse1_list.append(nse1)
        rmse1_list.append(rmse1)
        mae1_list.append(mae1)

        fed_fomo_model = nn.model.SequentialModel.load(fed_fomo_path)
        predict_y2 = fed_fomo_model.predict(val_x)
        predict_y2 = ds_test.local_rescale(predict_y2, variable='output')
        nse2, rmse2, mae2 = cal_nse_rmse_mae(val_y, predict_y2)
        nse2_list.append(nse2)
        rmse2_list.append(rmse2)
        mae2_list.append(mae2)


def eval_model(basin_id, flag, test_x, test_y, ds_test):
    # single_basin_data = None
    # ds_test = None
    # test_x = single_basin_data[0]
    # test_y = single_basin_data[1]
    local_lstm_path = './model/local_lstm_'+str(basin_id)+'.model'
    mutil_fune_path = './model/mutil_fune'+str(basin_id)+'.model'
    fed_hydro_path = './model/fed_hydro'+str(basin_id)+'.model'
    transfer_a_path = './model/tl_a_basin'+str(basin_id)+'.model'
    transfer_b_path = './model/tl_b_basin'+str(basin_id)+'.model'
    if flag == 1:
        fed_fomo_path = './model/fedfomo_10basins_spilt_rate0.7_client9_50epoch.model'
    else:
        fed_fomo_path = './model/fedfomo_10basins_spilt_rate0.7_client10_50epoch.model'
    local_lstm_model = nn.model.SequentialModel.load(local_lstm_path)
    mutil_fune_model = nn.model.SequentialModel.load(mutil_fune_path)
    fed_hydro_model = nn.model.SequentialModel.load(fed_hydro_path)
    transfer_a_model = nn.model.SequentialModel.load(transfer_a_path)
    transfer_b_model = nn.model.SequentialModel.load(transfer_b_path)
    fed_fomo_model = nn.model.SequentialModel.load(fed_fomo_path)
    model_list = [local_lstm_model, mutil_fune_model, fed_hydro_model, transfer_a_model, transfer_b_model, fed_fomo_model]
    nse_list = []
    rmse_list = []
    mae_list = []
    for model in model_list:
        predict_y = model.predict(test_x)
        predict_y = ds_test.local_rescale(predict_y, variable='output')
        nse, rmse, mae = cal_nse_rmse_mae(test_y, predict_y)
        nse_list.append(nse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        # np.savetxt()
    print("======================================basin:", basin_id, "========================================")
    print(nse_list)
    print(rmse_list)
    print(mae_list)
    return nse_list, rmse_list, mae_list


def eval_model_2years(basin_id, flag, test_x, test_y, ds_test):
    # single_basin_data = None
    # ds_test = None
    # test_x = single_basin_data[0]
    # test_y = single_basin_data[1]
    years = "_2years"
    local_lstm_path = './model/local_lstm_'+str(basin_id)+str(years)+'.model'
    mutil_fune_path = './model/mutil_fune'+str(basin_id)+str(years)+'.model'
    # fed_hydro_path = './model/fed_hydro'+str(basin_id)+'.model'
    transfer_a_path = './model/tl_a_basin'+str(basin_id)+str(years)+'.model'
    transfer_b_path = './model/tl_b_basin'+str(basin_id)+str(years)+'.model'
    if flag == 1:
        fed_fomo_path = './model/fedfomo_8basins_spilt_rate0.9_client7_70epoch.model'
    else:
        fed_fomo_path = './model/fedfomo_8basins_spilt_rate0.8_client8_70epoch.model'
    local_lstm_model = nn.model.SequentialModel.load(local_lstm_path)
    mutil_fune_model = nn.model.SequentialModel.load(mutil_fune_path)
    # fed_hydro_model = nn.model.SequentialModel.load(fed_hydro_path)
    transfer_a_model = nn.model.SequentialModel.load(transfer_a_path)
    transfer_b_model = nn.model.SequentialModel.load(transfer_b_path)
    fed_fomo_model = nn.model.SequentialModel.load(fed_fomo_path)
    model_list = [local_lstm_model, mutil_fune_model,  transfer_a_model, transfer_b_model, fed_fomo_model]
    # model_list = [local_lstm_model, mutil_fune_model, fed_hydro_model, transfer_a_model, transfer_b_model, fed_fomo_model]
    nse_list = []
    rmse_list = []
    mae_list = []
    for model in model_list:
        predict_y = model.predict(test_x)
        predict_y = ds_test.local_rescale(predict_y, variable='output')
        nse, rmse, mae = cal_nse_rmse_mae(test_y, predict_y)
        nse_list.append(nse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        # np.savetxt()
    print("======================================basin:", basin_id, "========================================")
    print(nse_list)
    print(rmse_list)
    print(mae_list)
    return nse_list, rmse_list, mae_list


def eval_model_2years2(basin_id, flag, test_x, test_y, ds_test):
    # single_basin_data = None
    # ds_test = None
    # test_x = single_basin_data[0]
    # test_y = single_basin_data[1]
    years = "_2years"
    local_lstm_path = './model/local_lstm_'+str(basin_id)+str(years)+'.model'
    mutil_fune_path = './model/mutil_fune'+str(basin_id)+str(years)+'2.model'
    # fed_hydro_path = './model/fed_hydro'+str(basin_id)+'.model'
    transfer_a_path = './model/tl_a_basin'+str(basin_id)+str(years)+'2.model'
    transfer_b_path = './model/tl_b_basin'+str(basin_id)+str(years)+'2.model'
    if flag == 1:
        fed_fomo_path = './model/fedfomo_8basins_spilt_rate0.9_client7_70epoch.model'
    else:
        fed_fomo_path = './model/fedfomo_8basins_spilt_rate0.8_client8_70epoch.model'
    local_lstm_model = nn.model.SequentialModel.load(local_lstm_path)
    mutil_fune_model = nn.model.SequentialModel.load(mutil_fune_path)
    # fed_hydro_model = nn.model.SequentialModel.load(fed_hydro_path)
    transfer_a_model = nn.model.SequentialModel.load(transfer_a_path)
    transfer_b_model = nn.model.SequentialModel.load(transfer_b_path)
    fed_fomo_model = nn.model.SequentialModel.load(fed_fomo_path)
    model_list = [local_lstm_model, mutil_fune_model,  transfer_a_model, transfer_b_model, fed_fomo_model]
    # model_list = [local_lstm_model, mutil_fune_model, fed_hydro_model, transfer_a_model, transfer_b_model, fed_fomo_model]
    nse_list = []
    rmse_list = []
    mae_list = []
    for model in model_list:
        predict_y = model.predict(test_x)
        predict_y = ds_test.local_rescale(predict_y, variable='output')
        nse, rmse, mae = cal_nse_rmse_mae(test_y, predict_y)
        nse_list.append(nse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        # np.savetxt()
    print("======================================basin:", basin_id, "========================================")
    print(nse_list)
    print(rmse_list)
    print(mae_list)
    return nse_list, rmse_list, mae_list


def get_date_range(test_date_range):
    the_start_date = test_date_range[0]
    the_end_date = test_date_range[1]
    test_date_range = {
        'train_date': {
            'start_date': pd.to_datetime("1980-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("1982-09-30", format="%Y-%m-%d")
        },
        'val_date': {
            'start_date': pd.to_datetime(the_start_date, format="%Y-%m-%d"),
            'end_date': pd.to_datetime(the_end_date, format="%Y-%m-%d")
        },
        'test_date': {
            'start_date': pd.to_datetime("2003-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2005-09-30", format="%Y-%m-%d")  # 01-03相对预测效果要差，属于与训练期间水文模型相差较大的年份
        },
    }
    return test_date_range


def correct_date_range_test():
    # a = ["1996-10-01", "1998-09-30"]
    # b = ["1998-10-01", "2000-09-30"]
    c = ["2000-10-01", "2002-09-30"]
    d = ["2002-10-01", "2004-09-30"]
    e = ["2004-10-01", "2006-09-30"]
    f = ["2006-10-01", "2008-09-30"]
    g = ["2008-10-01", "2010-09-30"]
    # h = ["20-10-01", "2010-09-30"]
    # i = ["2008-10-01", "2010-09-30"]
    date_range_list = []
    # date_range_list.append(a)
    # date_range_list.append(b)
    date_range_list.append(c)
    date_range_list.append(d)
    date_range_list.append(e)
    date_range_list.append(f)
    date_range_list.append(g)

    return date_range_list


def train_test_data(basin, the_date_range):
    train_x, train_y, val_x, val_y, ds_val, test_x, test_y, ds_test = get_sparse_data(basin, the_date_range)
    single_basin_data = []
    single_basin_data.append(train_x)
    single_basin_data.append(train_y)
    return single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test


def get_merge_data(rich_basin_ids, sparse_data_ids, rich_date_range, sparse_date_range):
    """得到Fedfomo训练测试数据"""
    train_data_list = []
    test_data_list = []
    ds_test_list = []
    for basin in rich_basin_ids:
        train_x, train_y, val_x, val_y, ds_val, test_x, test_y, ds_test = get_sparse_data(basin, rich_date_range)
        train_data = []
        test_data = []
        train_data.append(train_x)
        train_data.append(train_y)
        train_data_list.append(train_data)

        test_data.append(val_x)
        test_data.append(val_y)
        test_data_list.append(test_data)

        ds_test_list.append(ds_val)
    i = 0
    for basin in sparse_data_ids:
        train_x, train_y, val_x, val_y, ds_val, test_x, test_y, ds_test = get_sparse_data(basin, sparse_date_range[i])
        train_data = []
        test_data = []
        train_data.append(train_x)
        train_data.append(train_y)
        train_data_list.append(train_data)

        test_data.append(val_x)
        test_data.append(val_y)
        test_data_list.append(test_data)
        ds_test_list.append(ds_val)
        i += 1

    return train_data_list, test_data_list, ds_test_list


def fedsrp_test(rich_basin_ids, sparse_basin_ids, train_data_list, test_data_list, ds_test_list):
    round = 70
    eval_round = 2  # 5*5个epoch验证一次
    local_epoch = 2
    batch_size = 256
    target_basin_batch_size = 64
    num_basins = len(rich_basin_ids) + len(sparse_basin_ids)
    train_split = 0.7
    model_path = local_basic_model_name
    fed_fomo = FedFomo(model_path, round, eval_round, local_epoch, batch_size, target_basin_batch_size,
                       num_basins, train_data_list, test_data_list, ds_test_list, train_split, len(sparse_basin_ids))
    fed_fomo.run()


def get_rich_basin_res(basin_ids, flag):
    nse_list = []
    rmse_list = []
    mae_list = []
    i = 1
    for basin_id in basin_ids:
        single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin_id, rich_date_range)
        # model_path = ''
        if flag == 1:
            model_path = './model/fedfomo_8basins_spilt_rate0.8_client'+str(i)+'.model'
        else:
            model_path = './model/local_lstm_'+str(basin_id)+'.model'
        the_model = nn.model.SequentialModel.load(model_path)
        predict_y = the_model.predict(val_x)
        predict_y = ds_val.local_rescale(predict_y, variable='output')
        nse, rmse, mae = cal_nse_rmse_mae(val_y, predict_y)
        nse_list.append(nse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        i += 1
    print(nse_list)
    print(rmse_list)
    print(mae_list)


def plt_compare_res_with_30day(model_name, test_x, test_y, ds_test, eval_day_num):
    nse_list = []
    num = len(test_y) // eval_day_num
    n = len(test_y) - len(test_y) % eval_day_num
    test_x = test_x[:n, :, :]
    test_y = test_y[:n]
    cur_pos = 0
    model = nn.model.SequentialModel.load(model_name)
    for i in range(num):
        test_num_x = test_x[cur_pos:cur_pos+eval_day_num, :, :]
        test_num_y = test_y[cur_pos:cur_pos+eval_day_num]
        predict_y = model.predict(test_num_x)
        predict_y = ds_test.local_rescale(predict_y, variable='output')
        nse, rmse, mae = cal_nse_rmse_mae(test_num_y, predict_y)
        cur_pos += eval_day_num
        nse_list.append(nse)
    return nse_list


def plt_compare_res_with_30day_test():
    basin_id = '01047000'
    model_name1 = './model/fedfomo_8basins_spilt_rate0.7_client7_70epoch.model'
    model_name2 = './model/mutil_fune'+str(basin_id)+'.model'
    single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin_id, date_range1)
    eval_day_num = 90
    nse_list1 = plt_compare_res_with_30day(model_name1, val_x, val_y, ds_val, eval_day_num)
    nse_list2 = plt_compare_res_with_30day(model_name2, val_x, val_y, ds_val, eval_day_num)
    x = np.arange(1, len(nse_list1)+1)
    plt.plot(x, nse_list1, label='FedSRP')
    plt.plot(x, nse_list2, label='Mutil-Fune')
    plt.xlabel('Num')
    plt.ylabel('NSE')
    plt.legend()
    plt.show()


def plt_exp3(local_list, fedsrp_list):
    # 读取数据
    label_list = ['01013500', '01030500', '01031500', '01052500', '01057000']  # 横坐标刻度显示值
    x = np.arange(len(label_list))
    y1 = local_list
    y2 = fedsrp_list
    plt.figure(figsize=(10, 6))  # 设置画布的尺寸
    # plt.title('', fontsize=20)  # 标题，并设定字号大小
    plt.xlabel('流域编号', fontdict={'family': 'SimSun', 'size': 18})  # 设置x轴，并设定字号大小
    plt.ylabel('NSE', fontdict={'family': 'Times New Roman', 'size': 18})  # 设置y轴，并设定字号大小
    width_bar = 0.2
    # alpha：透明度；width：柱子的宽度；facecolor：柱子填充色；edgecolor：柱子轮廓色；lw：柱子轮廓的宽度；label：图例；
    plt.bar(x-width_bar/2, height=y1, width=width_bar, alpha=0.8, color="darkorange", label="Local-BLSTM")
    plt.bar(x + width_bar/2, height=y2, width=width_bar, color="#6488ea", label="FedSRP")

    plt.xticks(x, label_list, FontProperties='Times New Roman', size=18)
    plt.yticks(FontProperties='Times New Roman', size=18)
    plt.legend(prop={'family': 'Times New Roman', 'size': 12})  # 设置题注
    plt.show()  # 显示图像


def train_global_model():
    block_ids = [0, 1, 2, 3, 4, 5]
    epochs = [70]
    batch_size = 256
    basin_ids = ["01047000", "01054200"]
    single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin_ids[0], date_range1)
    train_x1 = single_basin_data[0]
    train_y1 = single_basin_data[1]

    single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin_ids[1], date_range1)
    train_x2 = single_basin_data[0]
    train_y2 = single_basin_data[1]
    sparse_train_x = np.concatenate((train_x1, train_x2), axis=0)
    sparse_train_y = np.concatenate((train_y1, train_y2), axis=0)
    for epoch in epochs:
        train_global_model_with_sparse(epoch, batch_size, block_ids, sparse_train_x, sparse_train_y)


def get_global_merge_data(rich_basin_ids, sparse_basin_ids, rich_date_range, sparse_date_range):
    rich_train_x = None
    rich_train_y = None
    rich_test_x = None
    rich_test_y = None
    for i in range(len(rich_basin_ids)):
        basin = rich_basin_ids[i]
        single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin, rich_date_range)
        train_x = single_basin_data[0]
        train_y = single_basin_data[1]
        if i == 0:
            rich_train_x = train_x
            rich_train_y = train_y
            rich_test_x = val_x
            rich_test_y = val_y
        else:
            rich_train_x = np.concatenate((rich_train_x, train_x), axis=0)
            rich_train_y = np.concatenate((rich_train_y, train_y), axis=0)
            rich_test_x = np.concatenate((rich_test_x, val_x), axis=0)
            rich_test_y = np.concatenate((rich_test_y, val_y), axis=0)

    sparse_train_x = None
    sparse_train_y = None
    sparse_test_x = None
    sparse_test_y = None
    for j in range(len(sparse_basin_ids)):
        basin = sparse_basin_ids[j]
        single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin, sparse_date_range[j])
        train_x = single_basin_data[0]
        train_y = single_basin_data[1]
        if j == 0:
            sparse_train_x = train_x
            sparse_train_y = train_y
            sparse_test_x = val_x
            sparse_test_y = val_y
        else:
            sparse_train_x = np.concatenate((sparse_train_x, train_x), axis=0)
            sparse_train_y = np.concatenate((sparse_train_y, train_y), axis=0)

            sparse_test_x = np.concatenate((sparse_test_x, val_x), axis=0)
            sparse_test_y = np.concatenate((sparse_test_y, val_y), axis=0)

    all_train_x = np.concatenate((rich_train_x, sparse_train_x), axis=0)
    all_train_y = np.concatenate((rich_train_y, rich_train_y), axis=0)

    all_test_x = np.concatenate((rich_test_x, sparse_test_x), axis=0)
    all_test_y = np.concatenate((rich_test_y, sparse_test_y), axis=0)

    return all_train_x, all_train_y, all_test_x, all_test_y


def train_global_model_merge(all_train_x, all_train_y, all_test_x, all_test_y, global_epoch, global_batch_size, sparse_basin_ids, basin_num):
    model = nn.model.SequentialModel.load(model_name)
    epoch_val_loss_list = []
    for i in range(1, global_epoch+1):
        model.fit(all_train_x, all_train_y, 1, global_batch_size)
        epoch_val_loss = model.evaluate_sum_loss(all_test_x, all_test_y, global_batch_size)
        epoch_val_loss = epoch_val_loss / all_test_x.shape[0]
        epoch_val_loss_list.append(epoch_val_loss)
    y_range = np.arange(1, global_epoch + 1)
    plt.plot(y_range, epoch_val_loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(
        './exp_res/exp2/global_model_loss/global_model' + str(epoch) + 'epochs_' + str(sparse_basin_ids[-1])+'.svg',
        format='svg')
    plt.close()
    # plt.show()
    model.save('./model/global_model_0414/global_model_'+str(basin_num)+'basins_'+str(global_epoch)+'epoch_2years'+str(sparse_basin_ids[-1])+'.model')


if __name__ == '__main__':
    """"""
    """7数据丰富 + 1稀缺"""
    rich_basin_ids1 = ['01013500', '01030500', '01031500', '01052500', '01055000', '01057000', '01054200']  # 01047000
    rich_basin_ids2 = ['01013500', '01030500', '01031500', '01052500', '01055000', '01057000', '01047000']  # 01054200
    rich_basin_ids3 = ['01013500', '01030500', '01031500', '01052500', '01057000', '01047000', '01054200']  # 01055000
    rich_basin_id_list = []
    rich_basin_id_list.append(rich_basin_ids1)
    rich_basin_id_list.append(rich_basin_ids2)
    rich_basin_id_list.append(rich_basin_ids3)

    sparse_basin_ids1 = ['01047000']
    sparse_basin_ids2 = ['01054200']
    sparse_basin_ids3 = ['01055000']
    sparse_basin_ids_list = []
    sparse_basin_ids_list.append(sparse_basin_ids1)
    sparse_basin_ids_list.append(sparse_basin_ids2)
    sparse_basin_ids_list.append(sparse_basin_ids3)

    rich_date_range = rich_date_range
    sparse_date_range = []
    # epochs = [50, 70, 100]
    epochs = [70]
    batch_size = 256
    for i in range(len(rich_basin_id_list)):
        if i == 0:
            sparse_date_range.append(date_range1)
        elif i == 1:
            sparse_date_range.append(date_range2)
        else:
            sparse_date_range.append(date_range3)
        rich_basin_ids = rich_basin_id_list[i]
        sparse_basin_ids = sparse_basin_ids_list[i]
        all_train_x, all_train_y, all_test_x, all_test_y = get_global_merge_data(rich_basin_ids, sparse_basin_ids,
                                                                                 rich_date_range, sparse_date_range)
        for epoch in epochs:
            train_global_model_merge(all_train_x, all_train_y, all_test_x, all_test_y, epoch, batch_size,
                                     sparse_basin_ids, len(rich_basin_ids1) + len(sparse_basin_ids1))
        sparse_date_range = []

    """后面的都不要了"""
    """global_model"""
    # epochs = [100, 150]
    # batch_sizes = [256, 256*2, 256*4]
    # for e in epochs:
    #         train_global_model(e, 256*2)
    # basin_id = "01022500"
    # basin_id = "01047000"
    # global_model_name1 = "./model/global_model_epoch50.model"
    # global_model_name2 = "./model/global_model_epoch100.model"
    # global_model_name3 = "./model/global_model_epoch150.model"
    # global_model_test(basin_id, global_model_name1)
    # global_model_test(basin_id, global_model_name2)
    # global_model_test(basin_id, global_model_name3)

    """local_lstm"""
    # basin_ids = ['01013500', '01030500', '01031500', '01052500', '01055000', '01057000']  # 01
    # # # # # # # basin_ids = ["02046000", "02051000"]  # 03
    # # # # # # # basin_ids = ["07056000", "07149000"]  # 11
    # basin_ids = ['01047000', '01054200']
    # local_basic_model_name = model_name
    # epoch = 70
    # # batch_size = 64
    # batch_size = 256
    # the_date_range = [date_range1, date_range2]
    # i = 0
    # for basin in basin_ids:
    #     train_x, train_y, val_x, val_y, ds_val, test_x, test_y, ds_test = get_sparse_data(basin, the_date_range[i])
    #     single_basin_data = []
    #     single_basin_data.append(train_x)
    #     single_basin_data.append(train_y)
    #     local_lstm(basin, local_basic_model_name, single_basin_data, val_x, val_y, ds_val, epoch, batch_size)
    #     i += 1
    # test_date_lsit = test_correct_date_range()
    # for test_date_range in test_date_lsit:
    #     for basin in basin_ids:
    #         the_date_range = get_date_range(test_date_range)
    #         train_x, train_y, val_x, val_y, ds_val, test_x, test_y, ds_test = get_sparse_data(basin, the_date_range)
    #         single_basin_data = []
    #         single_basin_data.append(train_x)
    #         single_basin_data.append(train_y)
    #         print("date:", test_date_range)
    #         local_lstm(basin, local_basic_model_name, single_basin_data, val_x, val_y, ds_val, epoch, batch_size)
    # the_date_range = [date_range1, date_range2]
    # i = 0
    # for basin in basin_ids:
    #     single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin, the_date_range[i])
    #     local_lstm(basin, local_basic_model_name, single_basin_data, val_x, val_y, ds_val, epoch, batch_size)
    #     i += 1

    """train global model"""
    # ['01013500', '01030500', '01031500', '01052500', '01055000', '01057000']
    # rich_basin_ids = ['01013500', '01030500', '01031500', '01052500', '01055000', '01057000']
    # rich_basin_ids = ['01013500', '01030500', '01031500', '01052500', '01055000', '01057000', '01054200'] # 0412
    # sparse_basin_ids = []
    # sparse_basin_ids = ['01047000', '01054200']


    # basin_ids = ['01013500', '01030500', '01031500', '01052500', '01055000', '01022500']
    """0412之前,  train global model"""
    # sparse_date_range.append(date_range1)
    # sparse_date_range.append(date_range2)
    # train_data_list, test_data_list, ds_test_list = get_merge_data(rich_basin_ids, sparse_basin_ids,
    # rich_date_range, sparse_date_range)
    # basin_ids = ['01013500', '01030500', '01031500', '01052500', '01055000', '01022500']
    # block_ids = [0, 1, 2, 3, 4, 5]
    # epochs = [70]
    # for epoch in epochs:
    #     train_global_model(epoch, 256, block_ids)
    # train_global_model()

    """mutil fune"""
    # global_model_name = './model/global_model_6basins_70epoch.model'
    # global_model_name = './model/global_model_6basins_70epoch_2years.model'
    # global_model_name = './model/global_model_0413/global_model_6basins_70epoch_2years.model'
    # # # basin_id = "01047000"
    # # # single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin_id, date_range1)
    # basin_ids = ["01047000", "01054200"]
    # i = 0
    # the_date_range = [date_range1, date_range2]
    # for basin_id in basin_ids:
    #     single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test =
    #     train_test_data(basin_id, the_date_range[i])
    #     epochs = [10]
    #     batch_size = 64
    #     for epoch in epochs:
    #         mutil_fune(global_model_name, basin_id, single_basin_data, epoch, batch_size, the_date_range[i])
    #     i += 1

    """transfer A/B"""
    # transfer_a
    # global_model_name = './model/global_model_6basins_70epoch.model'
    # global_model_name = './model/global_model_6basins_70epoch_2years.model'
    # basin_ids = ["01047000", "01054200"]
    # epoch = 10
    # batch_size = 64
    # the_date_range = [date_range1, date_range2]
    # # i = 0
    # # for basin in basin_ids:
    # #     single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin, the_date_range[i])
    # #     transfer_a(global_model_name, basin, single_basin_data, epoch, batch_size, val_x, val_y, ds_val)
    # #     i += 1
    #
    # # # # # transfer_b
    # i = 0
    # for basin in basin_ids:
    #     single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin, the_date_range[i])
    #     transfer_b(global_model_name, basin, single_basin_data, epoch, batch_size, val_x, val_y, ds_val)
    #     i += 1

    """fed_hydro"""
    # fed_model_name = './model/MODEL-fed_hydro_6basins_t30_210-N(0).model'
    # basin_ids = ["01047000", "01054200"]
    # epoch = 10
    # batch_size = 64
    # the_date_range = [date_range1, date_range2]
    # i = 0
    # for basin in basin_ids:
    #     single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin, the_date_range[i])
    #     fed_hydro(fed_model_name, single_basin_data, epoch, batch_size, val_x, val_y, ds_val)
    #     i += 1

    """eval model"""
    # basin_ids = ["01047000", "01054200"]
    # the_date_range = [date_range1, date_range2]
    # flag = [1, 2]
    # i = 0
    # for basin in basin_ids:
    #     single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin, the_date_range[i])
    #     eval_model(basin, flag[i], val_x, val_y, ds_val)
    #     print("下面是相似度大的测试集结果：====================================")
    #     eval_model(basin, flag[i], test_x, test_y, ds_test)
    #     i += 1

    """fed_srp"""
    # rich_basin_ids = ['01013500', '01030500', '01031500', '01052500', '01055000', '01057000']
    # # sparse_basin_ids = ['01047000', '01057000']
    # # rich_basin_ids = ['01013500',  '01030500', '01031500', '01052500',
    # # '01055000', '01022500', '01057000', '01073000']
    # # # rich_basin_ids = ['01013500', '01030500', '01031500', '01052500', '01055000', '01022500']
    # sparse_basin_ids = ['01047000', '01054200']
    # rich_date_range = rich_date_range
    # sparse_date_range = []
    # sparse_date_range.append(date_range1)
    # sparse_date_range.append(date_range2)
    # train_data_list, test_data_list, ds_test_list = get_merge_data(rich_basin_ids, sparse_basin_ids,
    # rich_date_range, sparse_date_range)
    # fedsrp_test(rich_basin_ids, sparse_basin_ids, train_data_list, test_data_list, ds_test_list)

    """fedsrp-数据丰富流域"""
    # basin_ids = []
    # rich_date_range = None
    # fed_srp_local_compare(basin_ids, rich_date_range)
    # rich_basin_ids = ['01013500', '01030500', '01031500', '01052500', '01055000', '01057000']
    # flag =1
    # get_rich_basin_res(rich_basin_ids, flag)
    # local_list = [0.794290733572722, 0.7837188686621781, 0.7629532070815569, 0.7209956949388628, 0.6393480516400291, 0.7379241653740303]
    # local_list = [0.794290733572722, 0.7837188686621781, 0.7629532070815569, 0.7209956949388628, 0.6393480516400291]
    # # fedsrp_list = [0.8412323511687417, 0.7910761482617809, 0.7781758500451543, 0.7666827312718688, 0.6385805633154007, 0.771826855489912]
    # fedsrp_list = [0.8412323511687417, 0.7910761482617809, 0.7781758500451543, 0.7666827312718688, 0.6385805633154007]
    # plt_exp3(local_list, fedsrp_list)

    # plt_compare_res_with_30day_test()

    """eval model 2 years"""
    # basin_ids = ["01047000", "01054200"]
    # the_date_range = [date_range1, date_range2]
    # flag = [1, 2]
    # i = 0
    # for basin in basin_ids:
    #     single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin, the_date_range[i])
    #     # eval_model_2years(basin, flag[i], val_x, val_y, ds_val)
    #     eval_model_2years2(basin, flag[i], val_x, val_y, ds_val)
    #     print("下面是相似度大的测试集结果：====================================")
    #     eval_model_2years2(basin, flag[i], test_x, test_y, ds_test)
    #     i += 1
