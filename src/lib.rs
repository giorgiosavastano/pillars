//! # Pillars
//!
//! `pillars` is a collection of algorithms implemented in Python and Rust.
//!
//! ## Highlights
//!
//! - Computation of EMD distance
//!

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::{exceptions, prelude::*, types::PyModule, wrap_pyfunction};

mod emd_classification;
mod matching;

#[pyfunction]
fn euclidean_rdist<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    // Convert PyReadonlyArray2 to ndarray::ArrayView2
    let x = x.as_array();
    let y = y.as_array();
    let z = emd_classification::euclidean_rdist_rust(x, y);

    let res = z.mapv(|elem| elem.into_inner());
    res.into_pyarray_bound(py)
}

#[pyfunction]
fn euclidean_rdist_parallel<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    let x = x.as_array();
    let y = y.as_array();
    let z = emd_classification::euclidean_rdist_par(x, y);
    let res = z.mapv(|elem| elem.into_inner());
    res.into_pyarray_bound(py)
}

#[pyfunction]
fn compute_emd<'py>(x: PyReadonlyArray2<'py, f64>, y: PyReadonlyArray2<'py, f64>) -> PyResult<f64> {
    let x = x.as_array();
    let y = y.as_array();
    let z = emd_classification::compute_emd_between_2dtensors(x, y);

    match z {
        Ok(z) => Ok(*z),
        Err(e) => Err(exceptions::PyTypeError::new_err(format!(
            "Failed to compute EMD distance: {}",
            e
        ))),
    }
}

#[pyfunction]
fn compute_emd_parallel<'py>(
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
) -> PyResult<f64> {
    let x = x.as_array();
    let y = y.as_array();
    let z = emd_classification::compute_emd_between_2dtensors_par(x, y);

    match z {
        Ok(z) => Ok(*z),
        Err(e) => Err(exceptions::PyTypeError::new_err(format!(
            "Failed to compute EMD distance: {}",
            e
        ))),
    }
}

#[pyfunction]
fn compute_emd_bulk<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray3<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let x = x.as_array();
    let y = y.as_array();
    let z = emd_classification::compute_emd_bulk(x, y);
    let z = z.mapv(|el| f64::from(el));
    z.into_pyarray_bound(py)
}

#[pyfunction]
fn compute_emd_bulk_par<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray3<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let x = x.as_array();
    let y = y.as_array();
    let z = emd_classification::compute_emd_bulk_par(x, y);
    let z = z.mapv(|el| f64::from(el));
    z.into_pyarray_bound(py)
}

#[pyfunction]
fn emd_classify<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray3<'py, f64>,
    n: usize,
) -> Bound<'py, PyArray1<usize>> {
    let x = x.as_array();
    let y = y.as_array();
    let z = emd_classification::classify_closest_n(x, y, n);
    z.into_pyarray_bound(py)
}

#[pyfunction]
fn emd_classify_bulk<'py>(
    py: Python<'py>,
    x: PyReadonlyArray3<'py, f64>,
    y: PyReadonlyArray3<'py, f64>,
    n: usize,
) -> Bound<'py, PyArray2<usize>> {
    let x = x.as_array();
    let y = y.as_array();
    let z = emd_classification::classify_closest_n_bulk(x, y, n);
    z.into_pyarray_bound(py)
}

#[pyfunction]
fn find_topk_with_tolerance<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    tolerance: f64,
    topk: usize,
) -> Bound<'py, PyArray2<i32>> {
    let x = x.as_array();
    let y = y.as_array();
    let z = matching::find_topk_with_tolerance(x, y, tolerance, topk);
    z.into_pyarray_bound(py)
}

/// This module is implemented in Rust.
#[pymodule]
fn pillars(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(euclidean_rdist, m)?)?;
    m.add_function(wrap_pyfunction!(euclidean_rdist_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(compute_emd, m)?)?;
    m.add_function(wrap_pyfunction!(compute_emd_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(compute_emd_bulk, m)?)?;
    m.add_function(wrap_pyfunction!(compute_emd_bulk_par, m)?)?;
    m.add_function(wrap_pyfunction!(emd_classify, m)?)?;
    m.add_function(wrap_pyfunction!(emd_classify_bulk, m)?)?;
    m.add_function(wrap_pyfunction!(find_topk_with_tolerance, m)?)?;
    Ok(())
}
