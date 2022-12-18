import numpy as np
from scipy.spatial.distance import cdist


if __name__ == '__main__':
    x = np.array([[1,2]])
    d = np.array([[1,3], [4,5], [4,6]])
    tt = cdist(x, d)

    idx = np.argmin(tt, axis=1)
    wtf = d[idx]
    wtf2 = tt[0, idx]

    omh = 3