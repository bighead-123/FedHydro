

def train_local_lstm(local_model, single_basin_data, epoch, batch_size):
    """仅用本地数据训练"""
    train_x = single_basin_data[0]
    train_y = single_basin_data[1]
    local_model.fit(train_x, train_y, epoch=epoch, batch_size=batch_size)
    return local_model