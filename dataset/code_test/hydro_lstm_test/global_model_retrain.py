import numpy as np
import os
import matplotlib.pyplot as plt
import nn
from nn.utils.nse_util import calc_nse
from dataset.code_test.utils_test.get_hydro_data import GetHydroData
from utils.constants import path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Training 01
file_path = path
# basin = '01030500'  # can be changed to any 8-digit basin id contained in the CAMELS 01 set
# basin = '01013500'  # can be changed to any 8-digit basin id contained in the CAMELS 01 set
# basin = '01031500'  # can be changed to any 8-digit basin id contained in the CAMELS 01 set
# basin = '07056000'  # can be changed to any 8-digit basin id contained in the CAMELS 01 set
# basin = '01054200'  # can be changed to any 8-digit basin id contained in the CAMELS 01 set
# basin = '01055000'  # can be changed to any 8-digit basin id contained in the CAMELS 01 set
hidden_size = 20  # Number of LSTM cells
dropout_rate = 0  # Dropout rate of the final fully connected Layer [0.0, 1.0]
learning_rate = 0.001  # Learning rate used to update the weights
sequence_length = 30  # Length of the meteorological record provided to the network


def global_model_retrain(basin_id, global_model_file_path):
    getHydroData = GetHydroData(basin_id, sequence_length)
    train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()
    # model = nn.model.Model.load('MODEL-fed_hydro_6basins_t30_210-N(0).model')
    # model = nn.model.Model.load('MODEL-fed_hydro_6basins_105-N(0).model')
    model = nn.model.Model.load(global_model_file_path)
    model.compile(nn.gradient_descent.ADAMOptimizer(alpha=learning_rate))
    print(model.summary())
    model_name_list = str(global_model_file_path).split('/')
    # model.fit(train_x, train_y, epoch=10, batch_size=256)
    # model.fit(train_x, train_y, epoch=15, batch_size=256)
    model.fit(train_x, train_y, epoch=10, batch_size=256)
    # model.fit(train_x, train_y, epoch=30, batch_size=256)
    # model = nn.model.Model.load('MODEL-fed_hydro_3basins-N(1).model')
    # model = nn.model.Model.load('MODEL-fed_hydro_3basins-N(2).model')
    model.save('./fed_model_retrain/'+str(model_name_list[-1]))
    ds_test = getHydroData.get_ds_test()
    predict_y = model.predict(test_x)
    predict_y = ds_test.local_rescale(predict_y, variable='output')
    # 计算NSE
    nse = calc_nse(test_y, predict_y)
    print("model-worker0,NSE:", nse)
    test_y = test_y[:1000]
    predict_y = predict_y[:1000]
    plt.subplot(2, 1, 1)  # 2行1列，第一张图
    plt.plot(list(range(len(predict_y))), predict_y, color='r', label='prediction')
    plt.plot(list(range(len(test_y))), test_y, color='b', label='origin')
    plt.title(f"basin:{basin_id}, NSE:{nse:.3f}")
    plt.xlabel('day number')
    plt.ylabel('discharge')
    plt.legend()
    plt.show()


if __name__=='__main__':
    basin = '07056000'
    global_model_file_path = './fed_model/MODEL-fed_hydro_unit11_12basins_t30_210-N(11).model'
    global_model_retrain(basin, global_model_file_path)



