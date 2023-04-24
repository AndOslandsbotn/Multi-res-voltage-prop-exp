from utilities.generate_toy_data import swiss_roll_domain, generate_2D_plane
from main_classes.epsilon_cover import build_epsilon_covers
import pickle
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def load_swissroll(filename):
    swissroll_data = np.load(os.path.join('Data', filename))
    return swissroll_data['swissroll'], swissroll_data['x'], swissroll_data['radius']

def save_to_file(epsilon_cover, filename):
    #Path(os.path.join('Data')).mkdir(parents=True, exist_ok=True)
    #for key in epsilon_cover:
    #    path = os.path.join('Data', filename + f'_lvl{key}' + '.mat')
    #    scipy.io.savemat(path, epsilon_cover[key])

    path = os.path.join('Data', filename + '.pcl')
    with open(path, 'wb') as f:
        pickle.dump(epsilon_cover, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    n = 100000
    filename = f'swissroll_n{n}.npz'
    swissroll, x, radius = load_swissroll(filename)
    epsilon_cover = build_epsilon_covers(swissroll)

    filename = f'swissroll_epscov_n{n}'
    save_to_file(epsilon_cover, filename)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(epsilon_cover[3]['centers'][:, 0],
               epsilon_cover[3]['centers'][:, 1],
               epsilon_cover[3]['centers'][:, 2], s=55, marker='x')
    ax.scatter(epsilon_cover[6]['centers'][:, 0],
               epsilon_cover[6]['centers'][:, 1],
               epsilon_cover[6]['centers'][:, 2])
    plt.show()




