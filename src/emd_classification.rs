use ndarray::prelude::*;
use ndarray::Zip;
use ordered_float::OrderedFloat;
use pathfinding::prelude::{kuhn_munkres_min, Matrix, MatrixFormatError};

fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    unsafe {
        indices.sort_by_key(|&i| data.get_unchecked(i));
    }
    indices
}

fn euclidean_distance(v1: &ArrayView1<f64>, v2: &ArrayView1<f64>) -> f64 {
    v1.iter()
        .zip(v2.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub fn euclidean_rdist_rust(x: ArrayView2<'_, f64>, y: ArrayView2<'_, f64>) -> Array2<f64> {
    let mut c = Array2::<f64>::zeros((x.nrows(), y.nrows()));
    for i in 0..x.nrows() {
        for j in 0..y.nrows() {
            unsafe {
                *c.uget_mut([i, j]) = euclidean_distance(&x.row(i), &y.row(j));
            }
        }
    }
    c
}

fn euclidean_rdist_row(x: &ArrayView1<'_, f64>, y: &ArrayView2<'_, f64>) -> Array1<f64> {
    let z = Zip::from(y.rows()).map_collect(|row| euclidean_distance(&row, &x));
    z
}

pub fn euclidean_rdist_par(x: ArrayView2<'_, f64>, y: ArrayView2<'_, f64>) -> Array2<f64> {
    let mut c = Array2::<f64>::zeros((x.nrows(), y.nrows()));
    Zip::from(x.rows())
        .and(c.rows_mut())
        .par_for_each(|row_x, mut row_c| row_c.assign(&euclidean_rdist_row(&row_x, &y)));
    c
}

pub fn emd_dist_serial(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
) -> Result<OrderedFloat<f64>, MatrixFormatError> {
    let c = euclidean_rdist_rust(x, y);
    let costs = c.mapv(|elem| OrderedFloat::from(elem));
    let weights = Matrix::from_vec(costs.nrows(), costs.ncols(), costs.into_raw_vec())?;
    let (emd_dist, _assignments) = kuhn_munkres_min(&weights);
    Ok(emd_dist)
}

fn compute_emd_bulk(x: ArrayView2<'_, f64>, y: ArrayView3<'_, f64>) -> Array1<OrderedFloat<f64>> {
    let mut c = Array1::<OrderedFloat<f64>>::zeros(y.shape()[0]);
    Zip::from(&mut c)
        .and(y.axis_iter(Axis(0)))
        .for_each(|c, mat_y| {
            *c = emd_dist_serial(mat_y, x).unwrap_or_else(|_e| {
                return OrderedFloat::from(999999.999);
            })
        });
    c
}

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
