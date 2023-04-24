from utilities.generate_toy_data import uniform_3d_sphere_section, place_landmarks

import os
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    nlm = 9
    n = 10000
    theta_max = 90
    phi_max = 180
    filename = f'sphere_n{n}'

    sphere = uniform_3d_sphere_section(n, theta_max=theta_max, phi_max = phi_max)
    radius = np.sqrt(sphere[:, 2] ** 2)

    landmarks = place_landmarks(nlm, theta_max, phi_max)

    Path(os.path.join('Data')).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join('Data', filename), sphere=sphere, radius=radius, landmarks=landmarks)

    plt.title('Epsilon cover')
    ax = plt.axes(projection='3d')
    ax.scatter(sphere[:, 0], sphere[:, 1], sphere[:, 2], c=radius)
    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], marker='v', s=155, color='r')
    plt.show()
