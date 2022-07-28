import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from pillars import compute_emd, emd_classify, emd_classify_bulk

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
	assert(len(emd_classes)==10)

def test_emd_classify_bulk():
	rng = np.random.default_rng()
	imgs_to_classify = rng.random((100, 17, 11))
	imgs_markers = rng.random((1000, 17, 11))
	emd_classes = emd_classify_bulk(imgs_to_classify, imgs_markers, 10)
	assert(emd_classes.shape==(imgs_to_classify.shape[0], 10))
