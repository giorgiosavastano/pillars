import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from pillars.emd_distance import (
    compute_earth_movers_distance_2d,
)  # Import your Rust-backed function


def compute_earth_mover_dist(first, second):
    """Compute earth's mover distance (EMD) between two data tensors using numpy and scipy."""
    emds = []
    for el in second:
        d = cdist(first, el)
        row_ind, col_ind = linear_sum_assignment(d)
        emd = d[row_ind, col_ind].sum()
        emds.append(emd)
    return emds


def setup_data():
    """Generates random data for benchmarking."""
    rng = np.random.default_rng()
    data1 = rng.random((50, 50))
    data2 = rng.random((10, 50, 50))
    return data1, data2


def test_rust_emd_benchmark(benchmark):
    """Benchmark the Rust-backed EMD calculation."""
    data1, data2 = setup_data()
    benchmark(compute_earth_movers_distance_2d, data1, data2, True)


def test_numpy_emd_benchmark(benchmark):
    """Benchmark the numpy/scipy EMD calculation."""
    data1, data2 = setup_data()
    benchmark(compute_earth_mover_dist, data1, data2)
