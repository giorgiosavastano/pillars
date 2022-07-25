import numpy as np
import itertools
from scipy.optimize import linear_sum_assignment
from pillars import rdist_parallel, rdist_serial


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

def emd_distance_bulk_map(cost_matrices):
	return np.asarray(list(map(emd_distance, cost_matrices)), dtype=float)

def emd_distance_bulk_iter(cost_matrices):
	return np.fromiter((emd_distance(xi) for xi in cost_matrices), float, count=cost_matrices.shape[0])

