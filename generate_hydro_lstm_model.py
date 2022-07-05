import nn


def gen_model(sequence_length):
    hidden_size = 20  # Number of LSTM cells
    dropout_rate = 0.2  # Dropout rate of the final fully connected Layer [0.0, 1.0]
    learning_rate = 1e-3  # Learning rate used to update the weights
    # sequence_length = 30  # Length of the meteorological record provided to the network
    # (-1, 30, 1)
    model = nn.model.SequentialModel(input_shape=(-1, sequence_length, 5))
    model.add(nn.layer.LSTM(n_in=5, units=hidden_size, nb_seq=sequence_length, return_sequence=True))
    model.add(nn.layer.LSTM(n_in=20, units=hidden_size, nb_seq=sequence_length))
    # model.add(nn.layer.BatchNorm())
    model.add(nn.layer.Dropout(drop_out_rate=dropout_rate))
    model.add(nn.layer.Dense(units=1))
    model.setup(nn.loss.MSELoss())
    model.compile(nn.gradient_descent.ADAMOptimizer(alpha=learning_rate))
    model.save("hydro_lstm_model-dropout.model")

    return model


def gen_model2(sequence_length):
    hidden_size = 128  # Number of LSTM cells
    dropout_rate = 0.2  # Dropout rate of the final fully connected Layer [0.0, 1.0]
    learning_rate = 1e-3  # Learning rate used to update the weights
    # sequence_length = 30  # Length of the meteorological record provided to the network
    # (-1, 30, 1)
    model = nn.model.SequentialModel(input_shape=(-1, sequence_length, 5))
    model.add(nn.layer.LSTM(n_in=5, units=hidden_size, nb_seq=sequence_length, return_sequence=False))
    # model.add(nn.layer.BatchNorm())
    model.add(nn.layer.Dropout(drop_out_rate=dropout_rate))
    model.add(nn.layer.Dense(units=1))
    model.setup(nn.loss.MSELoss())
    model.compile(nn.gradient_descent.ADAMOptimizer(alpha=learning_rate))
    model.save("hydro_lstm_model_dropout.model")

    return model


def gen_model3(sequence_length):
    hidden_size = 128  # Number of LSTM cells
    dropout_rate = 0.2  # Dropout rate of the final fully connected Layer [0.0, 1.0]
    learning_rate = 1e-3  # Learning rate used to update the weights
    # sequence_length = 30  # Length of the meteorological record provided to the network
    # (-1, 30, 1)
    model = nn.model.SequentialModel(input_shape=(-1, sequence_length, 5))
    model.add(nn.layer.LSTMDropout(n_in=5, units=hidden_size, nb_seq=sequence_length, recurrent_dropout=0.1, return_sequence=False))
    # model.add(nn.layer.BatchNorm())
    # model.add(nn.layer.Dropout(drop_out_rate=dropout_rate))
    model.add(nn.layer.Dense(units=1))
    model.setup(nn.loss.MSELoss())
    model.compile(nn.gradient_descent.ADAMOptimizer(alpha=learning_rate))
    model.save("hydro_lstm_model_dropout.model")

    return model


if __name__ == '__main__':
    sequence_length = 30
    gen_model2(sequence_length)
