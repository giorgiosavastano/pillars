# import numpy as np
# import itertools
# from scipy.optimize import linear_sum_assignment


# def emd_distance(cost_matrix):
# 	row_ind, col_ind = linear_sum_assignment(cost_matrix)
# 	return cost_matrix[row_ind, col_ind].sum()

# def emd_distance_bulk_map(cost_matrices):
# 	return np.asarray(list(map(emd_distance, cost_matrices)), dtype=float)

# def emd_distance_bulk_iter(cost_matrices):
# 	return np.fromiter((emd_distance(xi) for xi in cost_matrices), float, count=cost_matrices.shape[0])

