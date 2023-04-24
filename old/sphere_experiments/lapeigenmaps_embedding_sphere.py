from main_classes.laplacian_eigenmaps import LaplacianEigenmaps
import numpy as np
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def load_data(filename):
    data = np.load(os.path.join('Data', filename))
    return data['sphere'], data['radius']

if __name__ == '__main__':
    n = 10000

    results_folder = f'n{n}'
    results_filename = f'sphere_laplacian_eigenmaps_embedding_n{n}'

    filename = f'sphere_n{n}.npz'
    sphere, radius = load_data(filename)

    laplacianEigenmaps = LaplacianEigenmaps(results_folder, results_filename)
    laplacianEigenmaps.make_embedding(sphere)

    embedding = laplacianEigenmaps.get_embedding()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=radius)
    plt.show()

    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=radius)
    plt.show()
