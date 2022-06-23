import numpy as np


def cal_mae(obs: np.array, sim: np.array):
    """Calculate Nash-Sutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    # only consider time steps, where observations are available
    # sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    # obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    # sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    # obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    mae = np.mean(np.abs(obs-sim))
    return mae


def cal_mape(obs: np.array, sim: np.array):
    """Calculate Nash-Sutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    # only consider time steps, where observations are available
    sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    mape = np.mean(np.abs((obs-sim)/obs))*100
    return mape


if __name__ == '__main__':
    y = np.array([1, 2, 2, 3])
    y_hat = np.array([1, 2, 2, 4])
    print(cal_mape(y, y_hat))
