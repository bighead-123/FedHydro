import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import nn
from camelsTXT import CamelsTXT


"""Load hydro 01 from `path`"""
def load_data(path, kind):
    x_path = os.path.join(path, '%s_3basins_x.txt' % kind)
    y_path = os.path.join(path, '%s_3basins_y.txt' % kind)
    # 注意
    time_step = 30
    x = np.loadtxt(x_path, delimiter=' ').reshape((-1, time_step, 5))
    labels = np.loadtxt(y_path, delimiter=' ').reshape((-1, 1))
    return x, labels


# Training 01
basin = '01030500'  # can be changed to any 8-digit basin id contained in the CAMELS 01 set
hidden_size = 20  # Number of LSTM cells
dropout_rate = 0  # Dropout rate of the final fully connected Layer [0.0, 1.0]
learning_rate = 0.001  # Learning rate used to update the weights
sequence_length = 30  # Length of the meteorological record provided to the network

path = 'D:/河海大学/研究课题/水文预报/课题实验/Parallel-SGD/dataset/series_data/'
start_date = pd.to_datetime("1980-10-01", format="%Y-%m-%d")
end_date = pd.to_datetime("1995-09-30", format="%Y-%m-%d")
ds_train = CamelsTXT(basin, seq_length=sequence_length, period="train", dates=[start_date, end_date])
print("数据长度：", ds_train.__len__())
print("ds_train[0]，长度", ds_train[0], len(ds_train[0][0]))
# train_x = np.asarray(ds_train.x)
# train_y = np.asarray(ds_train.y)

means = ds_train.get_means()
stds = ds_train.get_stds()
start_date = pd.to_datetime("1995-10-01", format="%Y-%m-%d")
end_date = pd.to_datetime("2000-09-30", format="%Y-%m-%d")
ds_val = CamelsTXT(basin, seq_length=sequence_length, period="eval", dates=[start_date, end_date],
                     means=means, stds=stds)
val_x = np.asarray(ds_val.x)
val_y = np.asarray(ds_val.y)

# 测试数据
start_date = pd.to_datetime("2000-10-01", format="%Y-%m-%d")
end_date = pd.to_datetime("2010-09-30", format="%Y-%m-%d")
ds_test = CamelsTXT(basin, seq_length=sequence_length, period="eval", dates=[start_date, end_date],
                     means=means, stds=stds)
# test_x = np.asarray(ds_test.x)
# test_y = np.asarray(ds_test.y)


path = 'D:/河海大学/研究课题/水文预报/课题实验/Parallel-SGD/dataset/series_data/'
train_x, train_y = load_data(path, 'train')
test_x, test_y = load_data(path, 'test')
model = nn.model.SequentialModel(input_shape=(-1, train_x.shape[1], train_x.shape[2]))
model.add(nn.layer.LSTM(n_in=5, units=hidden_size, nb_seq=sequence_length))
model.add(nn.layer.Dense(units=1))
model.setup(nn.loss.MSELoss())
model.compile(nn.gradient_descent.ADAMOptimizer(alpha=learning_rate))
loss_history = model.fit(train_x, train_y, epoch=10,  batch_size=256)
model.save('self_lstm_test01.model')


predict_y = model.predict(test_x)
predict_y = ds_val.local_rescale(predict_y, variable='output')
# predict_y = ds_test.local_rescale(predict_y, variable='output')
test_y = test_y[:500]
predict_y = predict_y[:500]
print(loss_history.history)


epoch_loss = []
batch_num_per_epoch = loss_history.history[0][3]
i = 0
per_epoch_loss = 0
for batch_loss in loss_history.history:
    per_epoch_loss += batch_loss[-1]
    if i != 0 and i % batch_num_per_epoch == 0:
        epoch_loss.append(per_epoch_loss)
        per_epoch_loss = 0
    i = i + 1
print("epoch loss:", epoch_loss)
plt.subplot(2, 1, 1)  # 2行1列，第一张图
plt.plot(list(range(len(predict_y))), predict_y, color='r', label='prediction')
plt.plot(list(range(len(test_y))), test_y, color='b', label='origin')
plt.xlabel('epochs')
plt.ylabel('discharge')
plt.subplot(2, 1, 2)
plt.plot(list(range(len(epoch_loss))), epoch_loss, color='orange')
plt.xlabel('epochs')
plt.legend()
plt.show()




