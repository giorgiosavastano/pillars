use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use ndarray::Zip;
use ndarray::prelude::*;
use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[pymodule]
fn rust_cdist(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    fn euclidean_distance(v1: &ArrayView1<f64>, v2: &ArrayView1<f64>) -> f64 {
        v1.iter()
          .zip(v2.iter())
          .map(|(x,y)| (x - y).powi(2))
          .sum::<f64>()
          .sqrt()
    }


    fn euclidean_rdist_row_parallel(x: &ArrayView1<'_, f64>, y: &ArrayView2<'_, f64>) -> Array1<f64> {
        let z = Zip::from(y.rows()).map_collect(|row| euclidean_distance(&row, &x));
        z
    }

    fn euclidean_rdist_parallel(x: ArrayView2<'_, f64>, y: ArrayView2<'_, f64>) -> Array2<f64> {
        let mut c = Array2::<f64>::zeros((x.nrows(), y.nrows()));
        Zip::from(x.rows()).and(c.rows_mut()).par_for_each(|row_x, mut row_c| row_c.assign(&euclidean_rdist_row_parallel(&row_x, &y)));
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

    Ok(())
}