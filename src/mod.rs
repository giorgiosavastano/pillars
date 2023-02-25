use super::map_py_err;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray3};
use pyo3::prelude::*;

mod core;

pub type Triplet<'py, T, U> = (&'py PyArray1<T>, &'py PyArray1<T>, &'py PyArray1<U>);
pub type Quadruplet<'py, T> = (
    &'py PyArray1<T>,
    &'py PyArray1<T>,
    &'py PyArray1<T>,
    &'py PyArray1<T>,
);

#[pymodule]
pub fn pixel_wise_matching(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    macro_rules! pixel_wise_matching {
        ($name:ident, $t:ty, $u:ty, $v:ty) => {
            #[pyfunction]
            fn $name<'py>(
                py: Python<'py>,
                matching_array_left: PyReadonlyArray3<'py, $t>,
                differencing_array_left: PyReadonlyArray3<'py, $u>,
                matching_array_right: PyReadonlyArray3<'py, $t>,
                differencing_array_right: PyReadonlyArray3<'py, $u>,
                tol_match: $v,
                invalid: $t,
            ) -> PyResult<Triplet<'py, $t, $u>> {
                let matching_array_left = matching_array_left.as_array();
                let differencing_array_left = differencing_array_left.as_array();
                let matching_array_right = matching_array_right.as_array();
                let differencing_array_right = differencing_array_right.as_array();
                let (time_left, time_right, vals_out) = py
                    .allow_threads(|| {
                        core::pixel_wise_matching_with_tol(
                            &matching_array_left,
                            &differencing_array_left,
                            &matching_array_right,
                            &differencing_array_right,
                            tol_match,
                            invalid,
                        )
                    })
                    .map_err(|err| map_py_err(err.into()))?;
                Ok((
                    time_left.into_pyarray(py),
                    time_right.into_pyarray(py),
                    vals_out.into_pyarray(py),
                ))
            }

            m.add_function(wrap_pyfunction!($name, m)?)?;
        };
    }

    macro_rules! pixel_wise_matching_indexes {
        ($name:ident, $t:ty, $v:ty) => {
            #[pyfunction]
            fn $name<'py>(
                py: Python<'py>,
                matching_array_left: PyReadonlyArray3<'py, $t>,
                matching_array_right: PyReadonlyArray3<'py, $t>,
                tol_match: $v,
                invalid: $t,
            ) -> PyResult<Quadruplet<'py, usize>> {
                let matching_array_left = matching_array_left.as_array();
                let matching_array_right = matching_array_right.as_array();
                let (idxes_x, idxes_y, idxes_left, idxes_right) = py
                    .allow_threads(|| {
                        core::pixel_wise_matching_with_tol_indexes(
                            &matching_array_left,
                            &matching_array_right,
                            tol_match,
                            invalid,
                        )
                    })
                    .map_err(|err| map_py_err(err.into()))?;
                Ok((
                    idxes_x.into_pyarray(py),
                    idxes_y.into_pyarray(py),
                    idxes_left.into_pyarray(py),
                    idxes_right.into_pyarray(py),
                ))
            }

            m.add_function(wrap_pyfunction!($name, m)?)?;
        };
    }

    pixel_wise_matching!(pixel_wise_matching_f64_f64, f64, f64, f64);
    pixel_wise_matching!(pixel_wise_matching_f32_f32, f32, f32, f32);
    pixel_wise_matching!(pixel_wise_matching_f64_i64, i64, f64, i64);
    pixel_wise_matching!(pixel_wise_matching_f32_i64, i64, f64, i64);

    pixel_wise_matching_indexes!(pixel_wise_matching_indexes_f64, f64, f64);
    pixel_wise_matching_indexes!(pixel_wise_matching_indexes_f32, f32, f32);
    pixel_wise_matching_indexes!(pixel_wise_matching_indexes_i8, i8, i8);
    pixel_wise_matching_indexes!(pixel_wise_matching_indexes_i16, i16, i16);
    pixel_wise_matching_indexes!(pixel_wise_matching_indexes_i32, i32, i32);
    pixel_wise_matching_indexes!(pixel_wise_matching_indexes_i64, i64, i64);

    Ok(())
}
