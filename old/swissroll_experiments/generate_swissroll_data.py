from utilities.generate_toy_data import swiss_roll_domain, generate_2D_plane
from main_classes.epsilon_cover import build_epsilon_covers
import pickle
import os
from pathlib import Path
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    omega = 12
    n = 100000
    filename = f'swissroll_n{n}'
    x = generate_2D_plane(n)
    swissRoll, xcoord = swiss_roll_domain(x, omega)
    radius = np.sqrt(x[:, 0] ** 2)
    np.savez(os.path.join('Data', filename), swissroll=swissRoll, x=x, radius=radius)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(swissRoll[:, 0], swissRoll[:, 1], swissRoll[:, 2], c=radius)
    plt.show()




