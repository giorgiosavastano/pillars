import numpy as np
from scipy.optimize import linear_sum_assignment
from pillars import rdist


def emd_distance(x, y):

	cost_matrix = np.asarray(rdist(x.flatten(), y.flatten(), x.shape[1], y.shape[1]), dtype=float).reshape(x.shape[0], y.shape[0])
	row_ind, col_ind = linear_sum_assignment(cost_matrix)
	return cost_matrix[row_ind, col_ind].sum()
