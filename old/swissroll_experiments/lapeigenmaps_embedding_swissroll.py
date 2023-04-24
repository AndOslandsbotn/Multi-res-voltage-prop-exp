from main_classes.laplacian_eigenmaps import LaplacianEigenmaps
import numpy as np
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def load_swissroll(filename):
    swissroll_data = np.load(os.path.join('Data', filename))
    return swissroll_data['swissroll'], swissroll_data['x'], swissroll_data['radius']

if __name__ == '__main__':
    n = 5000
    source_lvl = 3
    results_folder = f'n{n}'
    results_filename = f'laplacian_eigenmaps_embedding_n{n}'

    filename = f'swissroll_n{n}.npz'
    swissroll, x, radius = load_swissroll(filename)

    laplacianEigenmaps = LaplacianEigenmaps(results_folder, results_filename)
    laplacianEigenmaps.make_embedding(swissroll)

    embedding = laplacianEigenmaps.get_embedding()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=radius)
    plt.show()

    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=radius)
    plt.show()
