import numpy as np
from scipy.spatial.distance import cdist
from pillars import rdist_parallel



def test_rdist():

	a = np.arange(100).reshape(10, 10)
	b = np.arange(200).reshape(20, 10)

	ncols_a = 10
	ncols_b = 10

	res_cdist = cdist(a, b)
	res_rdist = rdist_parallel(a, b)

	np.isclose(res_cdist, res_rdist)


def test_emd_distance():
	x = np.ones(shape=(11, 17)) * 11

	y = np.ones(shape=(11, 17)) * 11

	emd_dist = emd_distance(x, y)

	assert(emd_dist == 0.0)