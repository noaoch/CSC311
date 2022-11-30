import numpy as np


def l2_distance(a, b):
    """ Computes the Euclidean distance between a and b.
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B should be of same dimensionality.")

    aa = np.sum(a ** 2, axis=0)
    bb = np.sum(b ** 2, axis=0)
    ab = np.dot(a.T, b)

    return np.sqrt(aa[:, np.newaxis] + bb[np.newaxis, :] - 2 * ab)
