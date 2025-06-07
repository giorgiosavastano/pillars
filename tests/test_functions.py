import numpy as np
from pytest import approx, raises

# from netCDF4 import Dataset
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from pillars.emd_distance import (
    emd_classify,
    emd_classify_bulk,
    compute_euclidean_distance,
    compute_earth_movers_distance_2d,
)


def test_euclidean_distance_small_input():
    small_input1 = np.array([[0.1, 0.2], [0.3, 0.4]])
    small_input2 = np.array([[0.5, 0.6], [0.7, 0.8]])
    result = compute_euclidean_distance(small_input1, small_input2, parallel=False)
    expected = cdist(small_input1, small_input2)
    assert np.allclose(result, expected, rtol=1e-17)


def test_euclidean_distance_large_input():
    large_input1 = np.random.random((100, 100))
    large_input2 = np.random.random((100, 100))
    result = compute_euclidean_distance(large_input1, large_input2, parallel=True)
    expected = cdist(large_input1, large_input2)
    assert np.allclose(result, expected, rtol=1e-17)


def test_parallel_non_parallel_consistency():
    test_input1 = np.random.random((20, 20))
    test_input2 = np.random.random((20, 20))
    result_non_parallel = compute_euclidean_distance(
        test_input1, test_input2, parallel=False
    )
    result_parallel = compute_euclidean_distance(
        test_input1, test_input2, parallel=True
    )
    assert np.allclose(result_non_parallel, result_parallel, rtol=1e-17)


def test_performance():
    large_input1 = np.random.random((500, 500))
    large_input2 = np.random.random((500, 500))
    import time

    start_time = time.time()
    compute_euclidean_distance(large_input1, large_input2, parallel=True)
    elapsed_time_parallel = time.time() - start_time

    start_time = time.time()
    compute_euclidean_distance(large_input1, large_input2, parallel=False)
    elapsed_time_non_parallel = time.time() - start_time

    assert elapsed_time_parallel < elapsed_time_non_parallel


def test_emd_identical():
    identical_imgs = np.array([[1, 2], [3, 4]], dtype=np.float64)
    expected = 0
    # Assuming compute_emd expects a specific format, convert before passing
    actual = compute_earth_movers_distance_2d(identical_imgs, identical_imgs)
    assert np.allclose([expected], [actual], rtol=1e-17)


def test_emd_classify_single_marker():
    rng = np.random.default_rng()
    img_to_classify = rng.random((17, 11))
    single_img_marker = rng.random((2, 17, 11))  # Ensure there is more than one marker
    n_classes = 1  # Request fewer classes than markers
    emd_classes = emd_classify(img_to_classify, single_img_marker, n_classes)
    assert len(emd_classes) == n_classes


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
