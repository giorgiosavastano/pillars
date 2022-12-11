import numpy as np
import typing

from .pillars import (
    compute_emd,
    compute_emd_bulk,
    compute_emd_bulk_par,
    emd_classify,
    emd_classify_bulk,
    euclidean_rdist,
    euclidean_rdist_parallel,
    find_topk_with_tolerance,
)

__all__ = [
    "compute_emd",
    "compute_emd_bulk",
    "compute_emd_bulk_par",
    "emd_classify",
    "emd_classify_bulk",
    "euclidean_rdist",
    "euclidean_rdist_parallel",
    "find_topk_with_tolerance",
]


def compute_euclidean_distance(
    xa: np.ndarray, xb: np.ndarray, parallel: typing.Optional[bool] = True
) -> np.ndarray:
    """Compute Euclidean distance matrix between 2D arrays

    Args:
        xa (np.ndarray): 2D array
        xb (np.ndarray): 2D array
        parallel (typing.Optional[bool], optional): Parallel flag.
        Defaults to True.

    Returns:
        np.ndarray: Euclidea distance matrix
    """
    if parallel:
        return euclidean_rdist_parallel(xa, xb)
    else:
        return euclidean_rdist(xa, xb)


def compute_earth_movers_distance_2d(
    xa: np.ndarray, xb: np.ndarray, parallel: typing.Optional[bool] = True
) -> np.ndarray:
    """Compute Earth Movers Distance between 2D arrays

    Args:
        xa (np.ndarray): First array. It can be 2D or 3D.
        xb (np.ndarray): Second array. It can be 2D or 3D.
        parallel (typing.Optional[bool], optional): Parallel Flag.
        Defaults to True.

    Raises:
        ValueError: If shape of input array is not correct.

    Returns:
        np.ndarray: _description_
    """

    if len(xa.shape) == len(xb.shape) == 2:
        return np.asarray(compute_emd(xa, xb), dtype=float)

    elif len(xa.shape) == 2 and len(xb.shape) == 3:
        if parallel:
            return compute_emd_bulk_par(xa, xb)
        else:
            return compute_emd_bulk(xa, xb)

    elif len(xa.shape) == 3 and len(xb.shape) == 2:
        if parallel:
            return compute_emd_bulk_par(xb, xa)
        else:
            return compute_emd_bulk(xb, xa)

    else:
        raise ValueError("Dimensions of input arrays is not supported.")
