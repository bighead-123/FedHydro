
import numpy as np
import os
import nn

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

x_train = np.loadtxt('D:/河海大学/研究课题/水文预报/课题实验/Parallel-SGD/lstm_hydro_test/hydro_samples/01022500_x_train.txt', delimiter=' ').reshape((-1, 30, 5))
y_train = np.loadtxt('D:/河海大学/研究课题/水文预报/课题实验/Parallel-SGD/lstm_hydro_test/hydro_samples/01022500_y_train.txt', delimiter=' ').reshape((-1, 1))

model = nn.model.SequentialModel()
model.add(nn.layer.LSTM(n_in=5, units=20, nb_seq=30))
model.add(nn.layer.Dense(1))
model.setup(nn.loss.MSELoss())
model.compile(nn.gradient_descent.ADAMOptimizer(alpha=5e-4))
model.save('lstm_hydro.model')
print(model.summary())
# model.global_initialization()
# fit network
# train_y = np.reshape(train_y, (-1, 1))
# train_x = train_x[:2000, :]
# train_y = train_y[:2000, :]
model.fit(x_train, y_train, epoch=100,  batch_size=256)
# test_y = np.reshape(test_y, (-1, 1))
# # score = model.evaluate(test_X, test_y)
# predict = model.predict(test_X)
# test_y = np.reshape(test_y, (-1, 1))
# plt.figure(figsize=(24, 8))
# # plt.plot(list(range(len(train_y))), train_y, color='b', label='train_y')
# plt.plot(list(range(len(predict))), predict, color='r', label='prediction')
# plt.plot(list(range(len(test_y))), test_y, color='b', label='origin')
# plt.show()