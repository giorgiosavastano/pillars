use ndarray::{Zip, ArrayView1, Array3, Axis};
use std::path::PathBuf;


pub fn get_ddms_at_indices_ser(path: &PathBuf, variable_name: &String, indices: ArrayView1<usize>) -> Array3<f64> {
    let file = netcdf::open(path).unwrap();
    let var = &file.variable(variable_name).expect("Could not find variable name");
    let mut res = Array3::<f64>::zeros((indices.len(), 9, 5));
    Zip::from(res.axis_iter_mut(Axis(0))).and(&indices).for_each(|mut c, i| c.assign(&var.values::<f64>(Some(&[*i, 0, 0]), Some(&[1, 9, 5])).unwrap().into_shape((9, 5)).unwrap()));
    res
}

pub fn get_ddms_at_indices_par(path: &PathBuf, variable_name: &String, indices: ArrayView1<usize>) -> Array3<f64> {
    let file = netcdf::open(path).unwrap();
    let var = &file.variable(variable_name).expect("Could not find variable name");
    let mut res = Array3::<f64>::zeros((indices.len(), 9, 5));
    Zip::from(res.axis_iter_mut(Axis(0))).and(&indices).par_for_each(|mut c, i| c.assign(&var.values::<f64>(Some(&[*i, 0, 0]), Some(&[1, 9, 5])).unwrap().into_shape((9, 5)).unwrap()));
    res
}