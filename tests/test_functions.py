import numpy as np
from scipy.spatial.distance import cdist
from pillars import rdist_bulk, emd_bulk
from pillars.emdsa import rdist, emd_distance, emd_distance_bulk_dask


def test_rdist():

	a = np.linspace(0, 10, 100).reshape(10, 10)
	b = np.linspace(0, 10, 200).reshape(20, 10)

	res_cdist = cdist(a, b)
	res_rdist = rdist(a, b, parallel=True)

	np.isclose(res_cdist, res_rdist)


def test_emd_distance():
	x = np.ones(shape=(11, 17)) * 11

	y = np.ones(shape=(11, 17)) * 11

	res_rdist = rdist(x, y, parallel=False)

	emd_dist = emd_distance(res_rdist)

	assert(emd_dist == 0.0)

def test_rdist_bulk():
	x = np.random.rand(11, 17)
	y = np.random.rand(100, 11, 17)

	res = rdist_bulk(x, y)

	assert(res.shape == (y.shape[0], y.shape[1], x.shape[0]))

def test_emd_distance_bulk():

	x = np.random.rand(11, 17)
	y = np.random.rand(100, 11, 17)

	res = rdist_bulk(x, y)

	emd_vals = emd_distance_bulk_dask(res)

	assert(len(emd_vals)==res.shape[0])
