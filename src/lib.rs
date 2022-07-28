use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use ndarray::Zip;
use ndarray::prelude::*;
use pathfinding::prelude::{kuhn_munkres_min, Matrix};
use ordered_float::OrderedFloat;

use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[pymodule]
fn pillars(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
        let mut indices = (0..data.len()).collect::<Vec<_>>();
        unsafe{
            indices.sort_by_key(|&i| data.get_unchecked(i));
        }
        indices
    }

    fn euclidean_distance(v1: &ArrayView1<f64>, v2: &ArrayView1<f64>) -> f64 {
        v1.iter()
          .zip(v2.iter())
          .map(|(x,y)| (x - y).powi(2))
          .sum::<f64>()
          .sqrt()
    }

    fn euclidean_rdist_rust(x: ArrayView2<'_, f64>, y: ArrayView2<'_, f64>) -> Array2<f64> {
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

    fn emd_dist_serial(x: ArrayView2<'_, f64>, y: ArrayView2<'_, f64>) -> OrderedFloat<f64> {
        let c = euclidean_rdist_rust(x, y);
        let costs = c.mapv(|elem| OrderedFloat::from(elem));
        let weights = Matrix::from_vec(costs.nrows(), costs.ncols(), costs.into_raw_vec()).unwrap();
        let (emd_dist, _assignments) = kuhn_munkres_min(&weights);
        emd_dist
    }

    fn compute_emd_bulk(x: ArrayView2<'_, f64>, y: ArrayView3<'_, f64>) -> Array1<OrderedFloat<f64>> {
        let mut c = Array1::<OrderedFloat<f64>>::zeros(y.shape()[0]);
        Zip::from(&mut c).and(y.axis_iter(Axis(0))).for_each(|c, mat_y| *c = emd_dist_serial(mat_y, x));
        c
    }

    fn classify_closest_n(x: ArrayView2<'_, f64>, y: ArrayView3<'_, f64>, n: usize) -> Array1<usize> {
        let c = compute_emd_bulk(x, y);
        let res = argsort(&c.to_vec());
        assert!(n < res.len());
        unsafe{
            Array::from_vec(res.get_unchecked(0..n).to_vec())
        }
    }

    fn classify_closest_n_bulk(x: ArrayView3<'_, f64>, y: ArrayView3<'_, f64>, n: usize) -> Array2<usize> {
        let mut c = Array2::<usize>::zeros((x.shape()[0], n));
        Zip::from(c.rows_mut()).and(x.axis_iter(Axis(0))).par_for_each(|mut c, mat_x| c += &classify_closest_n(mat_x, y, n));
        c
    }


    #[pyfn(m)]
    fn compute_emd<'py>(
            x: PyReadonlyArray2<'py, f64>,
            y: PyReadonlyArray2<'py, f64>,
    ) -> f64 {
            let x = x.as_array();
            let y = y.as_array();
            let z = emd_dist_serial(x, y);
            f64::from(z)
    }

    #[pyfn(m)]
    fn emd_classify<'py>(
            py: Python<'py>,
            x: PyReadonlyArray2<'py, f64>,
            y: PyReadonlyArray3<'py, f64>,
            n: usize,
    ) -> &'py PyArray1<usize> {
            let x = x.as_array();
            let y = y.as_array();
            let z = classify_closest_n(x, y, n);
            z.into_pyarray(py)
    }

    #[pyfn(m)]
    fn emd_classify_bulk<'py>(
            py: Python<'py>,
            x: PyReadonlyArray3<'py, f64>,
            y: PyReadonlyArray3<'py, f64>,
            n: usize,
    ) -> &'py PyArray2<usize> {
            let x = x.as_array();
            let y = y.as_array();
            let z = classify_closest_n_bulk(x, y, n);
            z.into_pyarray(py)
    }

    Ok(())
}
