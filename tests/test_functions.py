import numpy as np

# from netCDF4 import Dataset
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from pillars import (
    compute_emd,
    emd_classify,
    emd_classify_bulk,
    euclidean_rdist,
    euclidean_rdist_parallel,
    # get_ddms_at_indices,
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
    actual = euclidean_rdist(imgs_test[0], imgs_test[1])
    actual_par = euclidean_rdist_parallel(imgs_test[0], imgs_test[1])
    assert np.allclose(expected, actual, rtol=1e-17)
    assert np.allclose(expected, actual_par, rtol=1e-17)


def test_emd_against_scipy():
    rng = np.random.default_rng()
    imgs_test = rng.random((2, 17, 11))
    expected = compute_earth_mover_dist(imgs_test[0], imgs_test[1])
    actual = compute_emd(imgs_test[0], imgs_test[1])
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


# def test_netcdf_ddms_indices():
#     root = "/Users/sysadmin/Develop/hrsm/error_propagation/data/"
#     prod = "L1B/gbrRCS/v01.01/2022/04/07/"
#     file = "test_file.nc"
#     path = f"{root}{prod}{file}"
#     var_name = "power_reflect"
#     indices = np.arange(0, 20, 2, dtype=np.uint64)
#     ddms_ser = get_ddms_at_indices(path, var_name, indices)
#     file_nc = Dataset(path, "r")
#     ddms_power = file_nc.variables["power_reflect"][:, :, :]
#     ddms_power = ddms_power[indices, :, :]
#     assert ddms_par.shape == (len(indices), 9, 5)
#     assert np.all(ddms_power == ddms_ser)
