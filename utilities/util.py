import numpy as np
from scipy.spatial.distance import cdist

def get_nn_indices(x, y, r):
    distances = cdist(x, y)
    condition = (distances <= r)
    idx_x, idx_y = np.where(condition == True)
    return idx_x, idx_y

def _dist(p1, p2):
    """
    :param p1: 1-dim np array
    :param p2: 1-dim np array
    :return: distance between p1 and p2
    """
    return np.sqrt(np.sum((p1-p2)**2, axis=0))

