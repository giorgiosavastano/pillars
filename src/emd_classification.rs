use ndarray::prelude::*;
use ndarray::Zip;
use ordered_float::OrderedFloat;
use pathfinding::prelude::{kuhn_munkres_min, Matrix, MatrixFormatError};

/// Represents an unusable or error state in distance calculations, initialized to infinity.
const BAD_VALUE: f64 = f64::INFINITY;

/// Sorts the indices of the elements in the provided slice in ascending order based on the elements themselves.
///
/// # Arguments
/// * `data` - A slice of data implementing the `Ord` trait.
///
/// # Returns
/// A vector of indices that, if used to index into `data`, will produce a sorted array.
fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    unsafe {
        indices.sort_by_key(|&i| data.get_unchecked(i));
    }
    indices
}

/// Calculates the Euclidean distance between two 1-dimensional arrays.
///
/// # Arguments
/// * `v1` - A 1-dimensional view of f64 data.
/// * `v2` - A 1-dimensional view of f64 data.
///
/// # Panics
/// Panics if the input arrays `v1` and `v2` have different lengths.
///
/// # Returns
/// The Euclidean distance as a floating-point number.
fn euclidean_distance(v1: &ArrayView1<f64>, v2: &ArrayView1<f64>) -> f64 {
    if v1.len() != v2.len() {
        panic!("Input arrays must have the same length");
    }
    Zip::from(v1)
        .and(v2)
        .map_collect(|&x, &y| (x - y).powi(2))
        .sum()
        .sqrt()
}

/// Computes the Euclidean distance between a single row and each row of a 2-dimensional array.
///
/// # Arguments
/// * `x` - A 1-dimensional view of a data row.
/// * `y` - A 2-dimensional array.
///
/// # Returns
/// A 1-dimensional array containing distances from `x` to each row in `y`.
fn euclidean_rdist_row(
    x: &ArrayView1<'_, f64>,
    y: &ArrayView2<'_, f64>,
) -> Array1<OrderedFloat<f64>> {
    if x.is_empty() || y.is_empty() {
        panic!("Input arrays must not be empty");
    }
    Zip::from(y.rows()).map_collect(|row| OrderedFloat::from(euclidean_distance(&row, &x)))
}

/// Computes the Euclidean distances between rows of two 2-dimensional data arrays synchronously.
///
/// # Arguments
/// * `x` - A 2-dimensional view of data arrays.
/// * `y` - A 2-dimensional view of data arrays.
///
/// # Returns
/// A 2-dimensional array where each element `(i, j)` is the distance between row `i` of `x` and row `j` of `y`.
pub fn euclidean_rdist_rust(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
) -> Array2<OrderedFloat<f64>> {
    let mut c = Array2::<OrderedFloat<f64>>::zeros((x.nrows(), y.nrows()));
    Zip::from(x.rows())
        .and(c.rows_mut())
        .for_each(|row_x, mut row_c| row_c.assign(&euclidean_rdist_row(&row_x, &y)));
    c
}

/// Similar to `euclidean_rdist_rust` but performs the computation in parallel.
///
/// # Arguments
/// * `x` - A 2-dimensional view of data arrays.
/// * `y` - A 2-dimensional view of data arrays.
///
/// # Returns
/// A 2-dimensional array where each element `(i, j)` is the distance between row `i` of `x` and row `j` of `y`.
pub fn euclidean_rdist_par(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
) -> Array2<OrderedFloat<f64>> {
    let mut c = Array2::<OrderedFloat<f64>>::zeros((x.nrows(), y.nrows()));
    Zip::from(x.rows())
        .and(c.rows_mut())
        .par_for_each(|row_x, mut row_c| row_c.assign(&euclidean_rdist_row(&row_x, &y)));
    c
}

/// Computes the Earth Movers Distance (EMD) between two 2-dimensional data tensors.
///
/// # Arguments
/// * `x` - A 2-dimensional view of f64 data tensors.
/// * `y` - A 2-dimensional view of f64 data tensors.
///
/// # Returns
/// A result containing the EMD as `OrderedFloat<f64>` or an error of type `MatrixFormatError`.
pub fn compute_emd_between_2dtensors(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
) -> Result<OrderedFloat<f64>, MatrixFormatError> {
    let costs = euclidean_rdist_rust(x, y);
    let weights = Matrix::from_vec(costs.nrows(), costs.ncols(), costs.into_raw_vec())?;
    let (emd_dist, _) = kuhn_munkres_min(&weights);
    Ok(emd_dist)
}

/// Computes the Earth Movers Distance (EMD) between two 2-dimensional data tensors.
///
/// # Arguments
/// * `x` - A 2-dimensional view of f64 data tensors.
/// * `y` - A 2-dimensional view of f64 data tensors.
///
/// # Returns
/// A result containing the EMD as `OrderedFloat<f64>` or an error of type `MatrixFormatError`.
pub fn compute_emd_between_2dtensors_par(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
) -> Result<OrderedFloat<f64>, MatrixFormatError> {
    let costs = euclidean_rdist_par(x, y);
    let weights = Matrix::from_vec(costs.nrows(), costs.ncols(), costs.into_raw_vec())?;
    let (emd_dist, _) = kuhn_munkres_min(&weights);
    Ok(emd_dist)
}

/// Computes the EMD between one 2D tensor and multiple 2D tensors contained in a 3D array, returning results for each computation.
///
/// # Arguments
/// * `x` - A 2-dimensional array view.
/// * `y` - A 3-dimensional array view.
///
/// # Returns
/// A 1-dimensional array where each element is the EMD from `x` to each 2D tensor in `y`.
pub fn compute_emd_bulk(
    x: ArrayView2<'_, f64>,
    y: ArrayView3<'_, f64>,
) -> Array1<OrderedFloat<f64>> {
    let mut c = Array1::<OrderedFloat<f64>>::zeros(y.shape()[0]);
    Zip::from(&mut c)
        .and(y.axis_iter(Axis(0)))
        .for_each(|c, mat_y| {
            *c = compute_emd_between_2dtensors(mat_y, x).unwrap_or_else(|err| {
                eprintln!("BAD_VALUE due to: {}", err);
                return OrderedFloat::from(BAD_VALUE);
            })
        });
    c
}

/// Similar to `compute_emd_bulk` but performs the computation in parallel.
///
/// # Arguments
/// * `x` - A 2-dimensional array view.
/// * `y` - A 3-dimensional array view.
///
/// # Returns
/// A 1-dimensional array where each element is the EMD from `x` to each 2D tensor in `y`.
pub fn compute_emd_bulk_par(
    x: ArrayView2<'_, f64>,
    y: ArrayView3<'_, f64>,
) -> Array1<OrderedFloat<f64>> {
    let mut c = Array1::<OrderedFloat<f64>>::zeros(y.shape()[0]);
    Zip::from(&mut c)
        .and(y.axis_iter(Axis(0)))
        .par_for_each(|c, mat_y| {
            *c = compute_emd_between_2dtensors(mat_y, x).unwrap_or_else(|err| {
                eprintln!("BAD_VALUE due to: {}", err);
                return OrderedFloat::from(BAD_VALUE);
            })
        });
    c
}

/// Identifies and returns the indices of the `n` closest tensors from `y` to `x` based on EMD.
///
/// # Arguments
/// * `x` - A 2-dimensional array view of f64.
/// * `y` - A 3-dimensional array view of f64 tensors.
/// * `n` - The number of closest tensors to identify.
///
/// # Returns
/// An array of indices corresponding to the closest tensors.
pub fn classify_closest_n(
    x: ArrayView2<'_, f64>,
    y: ArrayView3<'_, f64>,
    n: usize,
) -> Array1<usize> {
    let c = compute_emd_bulk(x, y);
    let res = argsort(&c.to_vec());
    assert!(n < res.len());
    unsafe { Array::from_vec(res.get_unchecked(0..n).to_vec()) }
}

/// Applies `classify_closest_n` for each tensor in `x` against all tensors in `y`, in parallel.
///
/// # Arguments
/// * `x` - A 3-dimensional array view of f64 tensors.
/// * `y` - A 3-dimensional array view of f64 tensors.
/// * `n` - The number of closest tensors to identify for each tensor in `x`.
///
/// # Returns
/// A 2-dimensional array where each row contains indices of the `n` closest tensors for each tensor in `x`.
pub fn classify_closest_n_bulk(
    x: ArrayView3<'_, f64>,
    y: ArrayView3<'_, f64>,
    n: usize,
) -> Array2<usize> {
    let mut c = Array2::<usize>::zeros((x.shape()[0], n));
    Zip::from(c.rows_mut())
        .and(x.axis_iter(Axis(0)))
        .par_for_each(|mut c, mat_x| c += &classify_closest_n(mat_x, y, n));
    c
}

#[cfg(test)]
mod emd_classification_tests {
    use super::*;

    #[test]
    fn argsort_test() {
        let a = Array1::from_vec(vec![100.125, 6.5489, 6.5488, 0.00, 77777.777]);
        let b = a.mapv(|elem| OrderedFloat::<f64>::from(elem));
        let c = argsort(&b.to_vec());

        assert_eq!(c, &[3, 2, 1, 0, 4]);
    }

    #[test]
    fn euclidean_rdist_rust_test() {
        let a = Array2::<f64>::zeros((1, 5));
        let b = Array2::<f64>::zeros((1, 5));
        let c = euclidean_rdist_rust(a.view(), b.view());
        assert_eq!(c.shape(), &[1, 1]);
    }
}
