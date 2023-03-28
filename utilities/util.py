from scipy.spatial.distance import cdist
import numpy as np
import pickle
import os
from time import perf_counter
import math

def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = perf_counter()
        result = func(*args, **kwargs)
        t2 = perf_counter()
        time = t2-t1
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result, time
    return wrap_func


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

def multi_dim_scaling(x):
    """ Make multi-dimensional scaling embedding of x
    Parameters
    ----------
    :param x: Coordinates as n x d numpy array, where n is number of training examples and d is the dimension
    :return: euclidean distance matrix n x n numpy array
    """
    centered = x - np.mean(x, axis = 0)
    u, sigma, vh = np.linalg.svd(centered)
    s_temp = np.zeros(len(x))
    s_temp[0:len(sigma)] = sigma[0:len(sigma)]
    sigma = s_temp
    x_mds= np.dot(u, np.diag(sigma))
    return x_mds

def load_pickle_data(data_folder, filename):
    with open(os.path.join(data_folder, filename +'.pcl'), 'rb') as pickle_file:
        content = pickle.load(pickle_file)
    return content

def cart2polar(X):
    R = np.sqrt(np.sum(np.square(X), axis=1))
    phi = np.arctan2(X[:, 1], X[:, 0])
    r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    theta = np.arctan2(r, X[:, 2])
    return R, theta, phi

def polar2cart(R, theta, phi):
    theta = (np.pi / 180) * theta
    phi = (np.pi / 180) * phi

    #phi = math.radians(phi)
    #theta = math.radians(theta)
    #phi = np.radians(phi)
    #theta = np.radians(theta)
    x = R*np.sin(theta) * np.cos(phi)
    y = R*np.sin(theta) * np.sin(phi)
    z = R*np.cos(theta)
    return x, y, z