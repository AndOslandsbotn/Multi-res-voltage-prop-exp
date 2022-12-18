
import numpy as np
from sklearn import datasets


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def generate_2D_plane(datasize):
    return np.random.uniform(0, 1, (datasize, 2))

def generate_swissRoll(N, D, eps):
    """Returns swiss roll and x coordinate"""
    swissRoll, xcoord = datasets.make_swiss_roll(n_samples=N, noise=eps, random_state=1)
    scale = np.std(swissRoll)
    return swissRoll, xcoord

def non_uniform_1d_experiment():
    n = 50
    density_source = 40*n
    density1 = 40*n
    density2 = 60*n
    density3 = 10*n
    density4 = 30*n
    density5 = 100*n

    x_nu = np.r_[np.linspace(0, 0.2, density1, endpoint=False), np.linspace(0.2, 0.4, density2, endpoint=False),
                 np.linspace(0.4, 0.6, density3, endpoint=False), np.linspace(0.6, 0.8, density4, endpoint=False),
                 np.linspace(0.8, 1, density5)]
    epsilon_cover_centers = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9])
    epsilon_cover_densities = np.array([density_source, density1, density2, density3, density4, density5])/len(x_nu)
    return x_nu.reshape(-1, 1), epsilon_cover_centers.reshape(-1, 1), epsilon_cover_densities