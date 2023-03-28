from utilities.kernels import select_kernel
import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix
from pathlib import Path
import os
from time import perf_counter
import json

class LaplacianEigenmaps():
    def __init__(self, results_folder, results_filename):
        self.results_folder = os.path.join('Results', results_folder)
        Path(self.results_folder).mkdir(parents=True, exist_ok=True)
        self.results_filename = results_filename
        self.embedding = None
        self.time_logg = {}

    def construct_weight_matrix(self, x):
        weight_matrix = select_kernel(x, bw=5, kernel_type='gaussian')
        diagonal_matrix = np.sum(weight_matrix, axis=1)
        diagonal_matrix = np.diag(diagonal_matrix)
        return weight_matrix, diagonal_matrix

    def construct_laplacian(self, x):
        n = len(x)
        weight_matrix, diagonal_matrix = self.construct_weight_matrix(x)

        Dinv = inv(csc_matrix(diagonal_matrix))
        weight_matarix_norm = Dinv.dot(weight_matrix)
        I = np.diag(np.ones(n))

        laplacian = np.dot(I, diagonal_matrix) - weight_matrix
        laplacian_norm = I - weight_matarix_norm
        return laplacian, laplacian_norm

    def make_embedding(self, x):
        start = perf_counter()
        L, Lnorm = self.construct_laplacian(x)
        sigma_norm, u_norm = np.linalg.eig(Lnorm)
        self.embedding = np.real(u_norm)[:, 1::]
        self.time_logg['exec_time_tot'] = perf_counter() - start

        self.save_embedding()
        self.save_time_logg()

    def get_embedding(self):
        return self.embedding

    def save_embedding(self):
        np.savez(os.path.join(self.results_folder, self.results_filename), embedding=self.embedding)

    def save_time_logg(self):
        with open(os.path.join(self.results_folder, 'time_logg_laplacian_eigenmaps'), "w") as outfile:
            json.dump(self.time_logg, outfile)


