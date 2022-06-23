import os
import matplotlib.pyplot as plt
import nn
from dataset.code_test.utils_test.get_hydro_data import GetHydroData
from nn.utils.nse_util import calc_nse
from utils.constants import path
# Training 01
file_path = path
basin = '01030500'  # can be changed to any 8-digit basin id contained in the CAMELS 01 set
# basin = '01013500'  # can be changed to any 8-digit basin id contained in the CAMELS 01 set
# basin = '01031500'  # can be changed to any 8-digit basin id contained in the CAMELS 01 set
# basin = '01052500'  # can be changed to any 8-digit basin id contained in the CAMELS 01 set
# basin = '01054200'  # can be changed to any 8-digit basin id contained in the CAMELS 01 set
# basin = '01055000'  # can be changed to any 8-digit basin id contained in the CAMELS 01 set

hidden_size = 20  # Number of LSTM cells
dropout_rate = 0  # Dropout rate of the final fully connected Layer [0.0, 1.0]
learning_rate = 0.001  # Learning rate used to update the weights
sequence_length = 30  # Length of the meteorological record provided to the network

getHydroData = GetHydroData(basin, sequence_length)
train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()
model = nn.model.Model.load('MODEL-fed_hydro_12basins_t30_210-N(0).model')
# model = nn.model.Model.load('lstm_hydro_01013500.model')
# model = nn.model.Model.load('lstm_hydro_01031500.model')
# model = nn.model.Model.load('lstm_hydro_01052500.model')
# model = nn.model.Model.load('lstm_hydro_01054200.model')
# model = nn.model.Model.load('lstm_hydro_01055000.model')
# model = nn.model.Model.load('MODEL-fed_hydro_3basins-N(1).model')
# model = nn.model.Model.load('MODEL-fed_hydro_3basins-N(2).model')
ds_test = getHydroData.get_ds_test()
predict_y = model.predict(test_x)
predict_y = ds_test.local_rescale(predict_y, variable='output')
# 计算NSE
nse = calc_nse(test_y, predict_y)
print("model-worker0,NSE:", nse)
test_y = test_y[:1000]
predict_y = predict_y[:1000]

# test_y = ds_test.reshape_discharge(test_y)
# predict_y = ds_test.reshape_discharge(predict_y)
plt.subplot(1, 1, 1)  # 2行1列，第一张图
plt.plot(list(range(len(predict_y))), predict_y, color='r', label='prediction')
plt.plot(list(range(len(test_y))), test_y, color='b', label='origin')
plt.title(f"basin:{basin}, NSE:{nse:.3f}")
plt.xlabel('day number')
plt.ylabel('discharge')
plt.legend()
# plt.savefig('fig_res/'+str(basin)+'.svg', format='svg')
plt.show()


# plt.subplot(2, 1, 2)
# plt.plot(list(range(len(epoch_loss))), epoch_loss, color='orange')
# plt.xlabel('epochs')
# print(loss_history.history)
# loss获取
# epoch_loss = []
# batch_num_per_epoch = loss_history.history[0][3]
# i = 0
# per_epoch_loss = 0
# for batch_loss in loss_history.history:
#     per_epoch_loss += batch_loss[-1]
#     if i != 0 and i % batch_num_per_epoch == 0:
#         epoch_loss.append(per_epoch_loss)
#         per_epoch_loss = 0
#     i = i + 1
# print("epoch loss:", epoch_loss)







