from utils.constants import path
import numpy as np
import os
import nn
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


x_train = np.loadtxt(path+'\\train_unit11_12basins_t30_x.txt', delimiter=' ').reshape((-1, 30, 5))
y_train = np.loadtxt(path+'\\train_unit11_12basins_t30_y.txt', delimiter=' ').reshape((-1, 1))

model = nn.model.SequentialModel()
model.add(nn.layer.LSTM(n_in=5, units=20, nb_seq=30, return_sequence=True))
model.add(nn.layer.LSTM(n_in=20, units=20, nb_seq=30))
model.add(nn.layer.Dense(1))
model.setup(nn.loss.MSELoss())
model.compile(nn.gradient_descent.ADAMOptimizer(alpha=5e-4))
model.save('lstm_hydro_unit11_12basins_t30_210.model')
print(model.summary())
# fit network
# train_y = np.reshape(train_y, (-1, 1))
# train_x = train_x[:2000, :]
# train_y = train_y[:2000, :]
model.fit(x_train, y_train, epoch=1,  batch_size=256)
# test_y = np.reshape(test_y, (-1, 1))
# # score = model.evaluate(test_X, test_y)
# predict = model.predict(test_X)
# test_y = np.reshape(test_y, (-1, 1))
# plt.figure(figsize=(24, 8))
# # plt.plot(list(range(len(train_y))), train_y, color='b', label='train_y')
# plt.plot(list(range(len(predict))), predict, color='r', label='prediction')
# plt.plot(list(range(len(test_y))), test_y, color='b', label='origin')
# plt.show()
