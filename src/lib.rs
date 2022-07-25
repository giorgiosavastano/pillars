use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use ndarray::Zip;
use ndarray::prelude::*;
use pathfinding::prelude::{kuhn_munkres, Matrix};
use ordered_float::OrderedFloat;

use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[pymodule]
fn pillars(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    fn euclidean_distance(v1: &ArrayView1<f64>, v2: &ArrayView1<f64>) -> f64 {
        v1.iter()
          .zip(v2.iter())
          .map(|(x,y)| (x - y).powi(2))
          .sum::<f64>()
          .sqrt()
    }

    fn euclidean_rdist_row(x: &ArrayView1<'_, f64>, y: &ArrayView2<'_, f64>) -> Array1<f64> {
        let z = Zip::from(y.rows()).map_collect(|row| euclidean_distance(&row, &x));
        z
    }

    fn euclidean_rdist_parallel(x: ArrayView2<'_, f64>, y: ArrayView2<'_, f64>) -> Array2<f64> {
        let mut c = Array2::<f64>::zeros((x.nrows(), y.nrows()));
        Zip::from(x.rows()).and(c.rows_mut()).par_for_each(|row_x, mut row_c| row_c.assign(&euclidean_rdist_row(&row_x, &y)));
        c
    }

    fn euclidean_rdist_serial(x: ArrayView2<'_, f64>, y: ArrayView2<'_, f64>) -> Array2<f64> {
        let mut c = Array2::<f64>::zeros((x.nrows(), y.nrows()));
        Zip::from(x.rows()).and(c.rows_mut()).for_each(|row_x, mut row_c| row_c.assign(&euclidean_rdist_row(&row_x, &y)));
        c
    }

    fn compute_euclidean_rdist_bulk(x: ArrayView2<'_, f64>, y: ArrayView3<'_, f64>) -> Array3<f64> {
        let mut c = Array3::<f64>::zeros((y.shape()[0], x.shape()[0], y.shape()[1]));
        Zip::from(y.axis_iter(Axis(0))).and(c.axis_iter_mut(Axis(0))).par_for_each(|mat_y, mut mat_c| mat_c.assign(&euclidean_rdist_serial(mat_y, x)));
        c
    }


    fn emd_dist_serial(x: ArrayView2<'_, f64>, y: ArrayView2<'_, f64>) -> f64 {
        let mut c = Array2::<f64>::zeros((x.nrows(), y.nrows()));
        Zip::from(x.rows()).and(c.rows_mut()).for_each(|row_x, mut row_c| row_c.assign(&euclidean_rdist_row(&row_x, &y)));

        let costs = c.mapv(|elem| OrderedFloat(elem));

        let weights = Matrix::from_vec(costs.nrows(), costs.ncols(), costs.into_raw_vec()).unwrap();
        let (emd_dist, assignments) = kuhn_munkres(&weights);
        emd_dist.0
    }

    fn compute_emd_bulk(x: ArrayView2<'_, f64>, y: ArrayView3<'_, f64>) -> Array1<f64> {
        let mut c = Array1::<f64>::zeros(y.shape()[0]);
        Zip::from(&mut c).and(y.axis_iter(Axis(0))).par_for_each(|mut c, mat_y| *c = emd_dist_serial(mat_y, x));
        c
    }
   
    #[pyfn(m)]
    fn rdist_parallel<'py>(
            py: Python<'py>,
            x: PyReadonlyArray2<'py, f64>,
            y: PyReadonlyArray2<'py, f64>,
    ) -> &'py PyArray2<f64> {
            let x = x.as_array();
            let y = y.as_array();
            let z = euclidean_rdist_parallel(x, y);
            z.into_pyarray(py)
    }

    #[pyfn(m)]
    fn rdist_serial<'py>(
            py: Python<'py>,
            x: PyReadonlyArray2<'py, f64>,
            y: PyReadonlyArray2<'py, f64>,
    ) -> &'py PyArray2<f64> {
            let x = x.as_array();
            let y = y.as_array();
            let z = euclidean_rdist_serial(x, y);
            z.into_pyarray(py)
    }

    #[pyfn(m)]
    fn rdist_bulk<'py>(
            py: Python<'py>,
            x: PyReadonlyArray2<'py, f64>,
            y: PyReadonlyArray3<'py, f64>,
    ) -> &'py PyArray3<f64> {
            let x = x.as_array();
            let y = y.as_array();
            let z = compute_euclidean_rdist_bulk(x, y);
            z.into_pyarray(py)
    }

    #[pyfn(m)]
    fn emd_bulk<'py>(
            py: Python<'py>,
            x: PyReadonlyArray2<'py, f64>,
            y: PyReadonlyArray3<'py, f64>,
    ) -> &'py PyArray1<f64> {
            let x = x.as_array();
            let y = y.as_array();
            let z = compute_emd_bulk(x, y);
            z.into_pyarray(py)
    }

    Ok(())
}
