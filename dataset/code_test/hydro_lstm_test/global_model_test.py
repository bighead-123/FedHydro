
import numpy as np
import os
import matplotlib.pyplot as plt
import nn
from nn.utils.nse_util import calc_nse
from dataset.code_test.utils_test.get_hydro_data import GetHydroData
from utils.constants import path
from generate_hydro_lstm_model import gen_model3
# Training 01
file_path = path
basin = '01030500'  # can be changed to any 8-digit basin id contained in the CAMELS 01 set
# basin = '01013500'  # can be changed to any 8-digit basin id contained in the CAMELS 01 set
# basin = '01031500'  # can be changed to any 8-digit basin id contained in the CAMELS 01 set
# basin = '01052500'
# basin = '01054200'
# basin = '01055000'


hidden_size = 20  # Number of LSTM cells
dropout_rate = 0  # Dropout rate of the final fully connected Layer [0.0, 1.0]
learning_rate = 0.001  # Learning rate used to update the weights
sequence_length = 30  # Length of the meteorological record provided to the network
# sequence_length = 30  # Length of the meteorological record provided to the network


getHydroData = GetHydroData(basin, sequence_length)
train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()

# model = nn.model.Model.load('MODEL-fed_hydro_3basins_t30_210-N(0).model')
# model = nn.model.Model.load('MODEL-fed_hydro_3basins-N(2).model')
# model = nn.model.Model.load('MODEL-fed_hydro_6basins_t10_210-N(5).model')
# model = nn.model.Model.load('MODEL-fed_hydro_9basins_t30_210-N(0).model')
model = gen_model3(sequence_length)
model.fit(train_x, train_y, epoch=120, batch_size=256)
# model = nn.model.Model.load   ('MODEL-fed_hydro_3basins-N(2).model')
ds_test = getHydroData.get_ds_test()
predict_y = model.predict(test_x)
predict_y = ds_test.local_rescale(predict_y, variable='output')
# 计算NSE
nse = calc_nse(predict_y, test_y)
print("model-worker0,NSE:", nse)
test_y = test_y[:500]
predict_y = predict_y[:500]
plt.subplot(2, 1, 1)  # 2行1列，第一张图
plt.plot(list(range(len(predict_y))), predict_y, color='r', label='prediction')
plt.plot(list(range(len(test_y))), test_y, color='b', label='origin')
plt.title(f"basin:{basin}, NSE:{nse:.3f}")
plt.xlabel('day number')
plt.ylabel('discharge')
plt.legend()
plt.show()







