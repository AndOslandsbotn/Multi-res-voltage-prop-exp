from utilities.util import multi_dim_scaling
import numpy as np
import os
import pickle

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def load_data(filename):
    data = np.load(os.path.join('Data', filename))
    return data['sphere'], data['radius']

def load_pickle_data(data_folder, filename):
    with open(os.path.join(data_folder, filename +'.pcl'), 'rb') as pickle_file:
        content = pickle.load(pickle_file)
    return content

if __name__ == '__main__':
    n = 10000
    embedding_lvl = 7

    # Load voltages
    filepath = os.path.join('Results', f'n{n}', f'sphere_voltage_embedding_n{n}_lvl{embedding_lvl}.npz')
    voltages = np.load(filepath)['voltages']

    # Load swissroll
    filename = f'sphere_n{n}.npz'
    sphere, radius = load_data(filename)

    # Load swissroll epsilon covers
    filename = f'sphere_epscov_n{n}'
    epsilon_cover = load_pickle_data('Data', filename)
    indices = epsilon_cover[embedding_lvl]['indices']

    voltages_mds = multi_dim_scaling(voltages.transpose())

    ##### Visualize
    fig = plt.figure()
    source_centers = epsilon_cover[3]['centers']
    source_indices = epsilon_cover[3]['indices']

    plt.title('Epsilon cover')
    ax = plt.axes(projection='3d')
    ax.scatter(source_centers[:, 0], source_centers[:, 1], source_centers[:, 2], marker='x', s=100, color='red')

    voltages  = np.sort(voltages, axis=1)
    plt.figure()
    for i in range(0, len(voltages)):
        plt.plot(voltages[i, :])

    fig = plt.figure()
    plt.title('Voltage embedding')
    ax = plt.axes(projection='3d')
    ax.scatter(voltages_mds[:, 0], voltages_mds[:, 1], voltages_mds[:, 2], c=radius[indices])

    plt.figure()
    plt.title('Voltage embedding')
    plt.scatter(voltages_mds[:, 0], voltages_mds[:, 1], c=radius[indices])

    plt.show()

    ### Laplacian eigenmaps
    filepath = os.path.join('Results', f'n{n}', f'sphere_laplacian_eigenmaps_embedding_n{n}.npz')
    laplacian_eigenmaps_embedding = np.load(filepath)['embedding']

    fig = plt.figure()
    plt.title('LE embedding')
    ax = plt.axes(projection='3d')
    ax.scatter(laplacian_eigenmaps_embedding[:, 0], laplacian_eigenmaps_embedding[:, 1], laplacian_eigenmaps_embedding[:, 2], c=radius)

    plt.title('LE embedding')
    plt.figure()
    plt.scatter(laplacian_eigenmaps_embedding[:, 0], laplacian_eigenmaps_embedding[:, 1], c=radius)
    plt.show()