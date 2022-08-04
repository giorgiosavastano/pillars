use ndarray::{Array3, ArrayView1, Axis, Zip};
use std::path::PathBuf;

pub fn get_ddms_at_indices_ser(
    path: &PathBuf,
    variable_name: String,
    indices: ArrayView1<usize>,
) -> Result<Array3<f64>, netcdf::error::Error> {
    let file = netcdf::open(path)?;

    let var = &file.variable(&variable_name);

    let var = match var {
        None => return Err(netcdf::error::Error::NotFound(variable_name)),
        Some(variable) => variable,
    };

    let mut res = Array3::<f64>::zeros((indices.len(), 9, 5));
    Zip::from(res.axis_iter_mut(Axis(0)))
        .and(&indices)
        .for_each(|mut c, i| {
            c.assign(
                &var.values::<f64>(Some(&[*i, 0, 0]), Some(&[1, 9, 5]))
                    .expect("Failed to get values at indices")
                    .into_shape((9, 5))
                    .expect("Failed to reshape ndarray"),
            )
        });
    Ok(res)
}
