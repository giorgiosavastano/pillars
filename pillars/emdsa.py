import numpy as np
from scipy.optimize import linear_sum_assignment
from pillars import rdist_parallel, rdist_serial
from dask import delayed, compute


def rdist(x, y, parallel=True):
	"""Compute euclidean distance between two matrices.
	
	Args:
		x (np.ndarray): 2-D array
		y (np.ndarray): 2-D array
		parallel (bool, optional): Parallel flag
	
	Returns:
		np.ndarray: 2-D array
	"""
	if parallel:
		cost_matrix = rdist_parallel(x, y)
	else:
		cost_matrix = rdist_serial(x, y)

	return cost_matrix


def emd_distance(cost_matrix):
	row_ind, col_ind = linear_sum_assignment(cost_matrix)
	return cost_matrix[row_ind, col_ind].sum()


def emd_distance_bulk_dask(cost_matrices):
	results = []
	for m in cost_matrices:
		y = delayed(emd_distance)(m)
		results.append(y)
	results = compute(*results)
	return results

