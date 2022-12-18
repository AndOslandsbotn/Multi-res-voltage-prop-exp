import numpy as np
from scipy.spatial.distance import cdist

def select_kernel(x, bw,  kernel_type):
    """Selects a kernel of kerne_type
    :param x: samples on which to construct the kernel
    :param bw: badnwidth of kernel
    """
    if kernel_type == 'radial':
        kernel = radial_kernel(x, bw)
    elif kernel_type == 'gaussian':
        kernel = gaussian_kernel(x, bw)
    return kernel

# Gaussian kernel
def gaussian_kernel(x, bandwidth, factor=None):
    bandwidth = bandwidth
    if factor != None:
        bandwidth = factor * bandwidth
    D = cdist(x, x, metric='sqeuclidean')
    D = (-1 / (2 * bandwidth ** 2)) * D
    return np.exp(D)

# Radial kernel
def radial_kernel(x, r):
    distances = cdist(x, x)
    W = (distances <= r).astype(int)
    return W
