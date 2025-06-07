import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from pillars.emd_distance import (
    compute_earth_movers_distance_2d,
)  # Import your Rust-backed function


def compute_earth_mover_dist(first, second):
    """Compute earth's mover distance (EMD) between two data tensors using numpy and scipy."""
    d = cdist(first, second)
    row_ind, col_ind = linear_sum_assignment(d)
    return d[row_ind, col_ind].sum()


def setup_data(rows, cols):
    """Generates random data for benchmarking."""
    rng = np.random.default_rng()
    data1 = rng.random((rows, cols), dtype=np.float64)
    data2 = rng.random((rows, cols), dtype=np.float64)
    return data1, data2


def test_rust_emd_benchmark_small(benchmark):
    """Benchmark the Rust-backed EMD calculation on small data."""
    data1, data2 = setup_data(17, 11)
    benchmark(compute_earth_movers_distance_2d, data1, data2, False)


def test_rust_emd_benchmark_medium(benchmark):
    """Benchmark the Rust-backed EMD calculation on medium data."""
    data1, data2 = setup_data(50, 50)
    benchmark(compute_earth_movers_distance_2d, data1, data2, False)


def test_rust_emd_benchmark_large(benchmark):
    """Benchmark the Rust-backed EMD calculation on large data."""
    data1, data2 = setup_data(100, 100)
    benchmark(compute_earth_movers_distance_2d, data1, data2, False)


def test_rust_emd_par_benchmark_small(benchmark):
    """Benchmark the Rust-backed EMD calculation with parallel processing on small data."""
    data1, data2 = setup_data(17, 11)
    benchmark(compute_earth_movers_distance_2d, data1, data2, True)


def test_rust_emd_par_benchmark_medium(benchmark):
    """Benchmark the Rust-backed EMD calculation with parallel processing on medium data."""
    data1, data2 = setup_data(50, 50)
    benchmark(compute_earth_movers_distance_2d, data1, data2, True)


def test_rust_emd_par_benchmark_large(benchmark):
    """Benchmark the Rust-backed EMD calculation with parallel processing on large data."""
    data1, data2 = setup_data(100, 100)
    benchmark(compute_earth_movers_distance_2d, data1, data2, True)


def test_numpy_emd_benchmark_small(benchmark):
    """Benchmark the numpy/scipy EMD calculation on small data."""
    data1, data2 = setup_data(17, 11)
    benchmark(compute_earth_mover_dist, data1, data2)


def test_numpy_emd_benchmark_medium(benchmark):
    """Benchmark the numpy/scipy EMD calculation on medium data."""
    data1, data2 = setup_data(50, 50)
    benchmark(compute_earth_mover_dist, data1, data2)


def test_numpy_emd_benchmark_large(benchmark):
    """Benchmark the numpy/scipy EMD calculation on large data."""
    data1, data2 = setup_data(100, 100)
    benchmark(compute_earth_mover_dist, data1, data2)
