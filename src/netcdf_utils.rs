use ndarray::{Array3, ArrayView1, Axis, Zip};
use std::path::PathBuf;

pub fn get_ddms_at_indices_ser(
    path: &PathBuf,
    variable_name: &String,
    indices: ArrayView1<usize>,
) -> Array3<f64> {
    let file = netcdf::open(path).expect("Failed to open file");
    let var = &file
        .variable(variable_name)
        .expect("Failed to find variable name");
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
    res
}
