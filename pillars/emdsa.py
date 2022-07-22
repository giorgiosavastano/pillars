import numpy as np
from scipy.optimize import linear_sum_assignment
from pillars import rdist_parallel


def emd_distance(x, y):

	cost_matrix = rdist_parallel(x, y)
	row_ind, col_ind = linear_sum_assignment(cost_matrix)
	return cost_matrix[row_ind, col_ind].sum()
