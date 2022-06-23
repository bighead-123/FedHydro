import numpy as np


def cal_rmse(obs: np.array, sim: np.array):
    # only consider time steps, where observations are available
    # sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    # obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    # sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    # obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    rmse = np.sqrt(np.mean(np.square(sim - obs)))
    return rmse
