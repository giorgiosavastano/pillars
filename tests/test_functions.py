import numpy as np

# from netCDF4 import Dataset
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from pillars import (
    emd_classify,
    emd_classify_bulk,
    compute_euclidean_distance,
    compute_earth_movers_distance_2d,
)


def compute_earth_mover_dist(first, second):
    """
    Compute earth's mover distance (EMD) between two data tensors.
    Parameters
    ----------
    first : np.ndarray
        First data array
    second : np.ndarray
        Second data array
    Returns
    ----------
    emd_val : float
        EMD distance between the two arrays
    """
    d = cdist(first, second)
    row_ind, col_ind = linear_sum_assignment(d)
    emd_val = d[row_ind, col_ind].sum()
    return emd_val


def test_rdist_against_scipy():
    rng = np.random.default_rng()
    imgs_test = rng.random((2, 17, 11))
    expected = cdist(imgs_test[0], imgs_test[1])
    actual = compute_euclidean_distance(imgs_test[0], imgs_test[1], parallel=False)
    actual_par = compute_euclidean_distance(imgs_test[0], imgs_test[1], parallel=True)
    assert np.allclose(expected, actual, rtol=1e-17)
    assert np.allclose(expected, actual_par, rtol=1e-17)


def test_emd_against_scipy():
    rng = np.random.default_rng()
    imgs_test = rng.random((2, 17, 11))
    expected = compute_earth_mover_dist(imgs_test[0], imgs_test[1])
    actual = compute_earth_movers_distance_2d(imgs_test[0], imgs_test[1])
    assert np.allclose(expected, actual, rtol=1e-17)


def test_emd_classify():
    rng = np.random.default_rng()
    img_to_classify = rng.random((17, 11))
    imgs_markers = rng.random((100, 17, 11))
    emd_classes = emd_classify(img_to_classify, imgs_markers, 10)
    assert len(emd_classes) == 10


def test_emd_classify_bulk():
    rng = np.random.default_rng()
    imgs_to_classify = rng.random((100, 17, 11))
    imgs_markers = rng.random((1000, 17, 11))
    emd_classes = emd_classify_bulk(imgs_to_classify, imgs_markers, 10)
    assert emd_classes.shape == (imgs_to_classify.shape[0], 10)
