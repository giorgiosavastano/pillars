use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

mod emd_classification;
mod netcdf_utils;

#[pymodule]
fn pillars(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn euclidean_rdist<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray2<'py, f64>,
    ) -> &'py PyArray2<f64> {
        let x = x.as_array();
        let y = y.as_array();
        let z = emd_classification::euclidean_rdist_rust(x, y);
        z.into_pyarray(py)
    }

    #[pyfn(m)]
    fn euclidean_rdist_parallel<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray2<'py, f64>,
    ) -> &'py PyArray2<f64> {
        let x = x.as_array();
        let y = y.as_array();
        let z = emd_classification::euclidean_rdist_par(x, y);
        z.into_pyarray(py)
    }

    #[pyfn(m)]
    fn compute_emd<'py>(x: PyReadonlyArray2<'py, f64>, y: PyReadonlyArray2<'py, f64>) -> f64 {
        let x = x.as_array();
        let y = y.as_array();
        let z = emd_classification::emd_dist_serial(x, y);
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
        let z = emd_classification::classify_closest_n(x, y, n);
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
        let z = emd_classification::classify_closest_n_bulk(x, y, n);
        z.into_pyarray(py)
    }

    #[pyfn(m)]
    fn get_ddms_at_indices<'py>(
        py: Python<'py>,
        path: std::path::PathBuf,
        var_name: String,
        x: PyReadonlyArray1<'py, usize>,
    ) -> &'py PyArray3<f64> {
        let x = x.as_array();
        let ddms = netcdf_utils::get_ddms_at_indices_ser(&path, &var_name, x);
        ddms.into_pyarray(py)
    }

    Ok(())
}
