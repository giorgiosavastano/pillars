import numpy as np
from pillars import rdist_bulk, emd_bulk, emd_classify, emd_classify_bulk


def test_rdist_bulk():
	x = np.random.rand(11, 17)
	y = np.random.rand(100, 11, 17)

	res = rdist_bulk(x, y)

	assert(res.shape == (y.shape[0], y.shape[1], x.shape[0]))

def test_emd_distance_bulk():

	x = np.random.rand(11, 17)
	y = np.random.rand(100, 11, 17)

	emd_vals = emd_bulk(x, y)

	assert(len(emd_vals)==y.shape[0])


def test_emd_classify():

	x = np.random.rand(11, 17)
	y = np.random.rand(100, 11, 17)

	emd_classes = emd_classify(x, y, 10)

	assert(len(emd_classes)==10)

def test_emd_classify_bulk():

	x = np.random.rand(100, 11, 17)
	y = np.random.rand(1000, 11, 17)

	emd_classes = emd_classify_bulk(x, y, 10)

	assert(emd_classes.shape==(x.shape[0], 10))