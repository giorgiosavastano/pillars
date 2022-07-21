import numpy as np
from scipy.spatial.distance import cdist
from pillars import sum_as_string, rdist
from pillars.emdsa import emd_distance


def test_sum_as_string():

	a = 10
	b = 20

	res_str = sum_as_string(a, b)

	assert(type(res_str) == str)


def test_rdist():

	a = np.arange(100).reshape(10, 10)
	b = np.arange(200).reshape(20, 10)

	ncols_a = 10
	ncols_b = 10

	res_cdist = cdist(a, b)
	res_rdist = rdist(a.flatten(), b.flatten(), ncols_a, ncols_b)

	assert(len(res_rdist) == 200)
	np.isclose(res_cdist.flatten(), res_rdist)


def test_emd_distance():
	x = np.ones(shape=(11, 17)) * 11

	y = np.ones(shape=(11, 17)) * 11

	emd_dist = emd_distance(x, y)

	assert(emd_dist == 0.0)