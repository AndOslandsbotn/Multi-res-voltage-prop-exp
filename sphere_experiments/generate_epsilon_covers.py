from utilities.generate_toy_data import swiss_roll_domain, generate_2D_plane
from main_classes.epsilon_cover import build_epsilon_covers
import pickle
import os
from scipy.spatial.distance import cdist

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def save_to_file(epsilon_cover, filename):
    path = os.path.join('Data', filename + '.pcl')
    with open(path, 'wb') as f:
        pickle.dump(epsilon_cover, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    n = 10000
    filename = f'sphere_n{n}.npz'
    data = np.load(os.path.join('Data', filename))
    sphere, radius, landmarks = data['sphere'], data['radius'], data['landmarks']

    epsilon_cover = build_epsilon_covers(sphere)
    distances = cdist(landmarks, epsilon_cover[3]['centers'])
    idx = np.argmin(distances, axis=1)

    #epsilon_cover[3]['centers'] = epsilon_cover[3]['centers'][idx]
    #epsilon_cover[3]['densities'] = epsilon_cover[3]['densities'][idx]
    #epsilon_cover[3]['indices'] = epsilon_cover[3]['indices'][idx]

    filename = f'sphere_epscov_n{n}'
    save_to_file(epsilon_cover, filename)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(epsilon_cover[3]['centers'][:, 0],
               epsilon_cover[3]['centers'][:, 1],
               epsilon_cover[3]['centers'][:, 2], s=155, marker='x')
    ax.scatter(epsilon_cover[6]['centers'][:, 0],
               epsilon_cover[6]['centers'][:, 1],
               epsilon_cover[6]['centers'][:, 2])
    plt.show()




