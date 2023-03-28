from utilities.util import polar2cart
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def swiss_roll_domain(x, omega):
    """Returns swiss roll and x coordinate"""
    swissRoll = np.array([x[:, 0]*np.cos(omega*x[:, 0]), x[:, 1], x[:, 0]*np.sin(omega*x[:, 0])]).transpose()
    xcoord = x[:, 0]
    return swissRoll, xcoord

def uniform_3d_sphere_section(N, theta_max=360, phi_max = 180):
    u = np.random.random(N)
    v = np.random.random(N)

    c1 = 360 / theta_max
    c2 = 180 / phi_max

    theta = 360 / c1 * u  # Angles in degrees
    phi = (180 /np.pi) * np.arccos(2*v - 1) / c2  # Angles in degrees

    x, y, z = polar2cart(1, theta, phi)
    return np.array([x, y, z]).transpose()

def generate_2D_plane(datasize):
    return np.random.uniform(0, 1, (datasize, 2))

def non_uniform_1d_experiment():
    n = 100
    density_source = 40*n
    density1 = 40*n
    density2 = 60*n
    density3 = 10*n
    density4 = 30*n
    density5 = 100*n

    #density_source = 60*n
    #density1 = 60*n
    #density2 = 60*n
    #density3 = 60*n
    #density4 = 60*n
    #density5 = 60*n

    x_nu = np.r_[np.linspace(0, 0.2, density1, endpoint=False), np.linspace(0.2, 0.4, density2, endpoint=False),
                 np.linspace(0.4, 0.6, density3, endpoint=False), np.linspace(0.6, 0.8, density4, endpoint=False),
                 np.linspace(0.8, 1, density5)]
    epsilon_cover_centers = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9])
    epsilon_cover_densities = np.array([density_source, density1, density2, density3, density4, density5])/len(x_nu)
    return x_nu.reshape(-1, 1), epsilon_cover_centers.reshape(-1, 1), epsilon_cover_densities

def place_landmarks(nlm, theta_max, phi_max):
    # Place landmarks on sphere
    landmarks = []

    x, y, z = polar2cart(R=1, theta=5, phi=phi_max/2)
    landmarks.append(np.array([x, y, z]))
    x, y, z = polar2cart(R=1, theta=theta_max-5, phi=5)
    landmarks.append(np.array([x, y, z]))
    x, y, z = polar2cart(R=1, theta=theta_max-5, phi=phi_max-5)
    landmarks.append(np.array([x, y, z]))

    if nlm >= 5:
        x, y, z = polar2cart(R=1, theta=theta_max/2, phi=5)
        landmarks.append(np.array([x, y, z]))
        x, y, z = polar2cart(R=1, theta=theta_max/2, phi=phi_max)
        landmarks.append(np.array([x, y, z]))

    if nlm >= 7:
        x, y, z = polar2cart(R=1, theta=theta_max, phi=phi_max/2)
        landmarks.append(np.array([x, y, z]))
        x, y, z = polar2cart(R=1, theta=theta_max / 2, phi=phi_max / 2)
        landmarks.append(np.array([x, y, z]))

    if nlm >= 9:
        x, y, z = polar2cart(R=1, theta= theta_max, phi=phi_max/4)
        landmarks.append(np.array([x, y, z]))
        x, y, z = polar2cart(R=1, theta= theta_max , phi=3*phi_max/4)
        landmarks.append(np.array([x, y, z]))
    return np.array(landmarks)