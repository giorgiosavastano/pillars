use ndarray::prelude::*;
use ndarray::ErrorKind;
use ndarray::ShapeError;
use ndarray::Zip;

type Triplet<T, U> = (Vec<T>, Vec<T>, Vec<U>);
type Quadruplet<T> = (Vec<T>, Vec<T>, Vec<T>, Vec<T>);

pub trait Absolute {
    fn abs(&self) -> Self;
}

impl<T> Absolute for T
where
    T: Default + PartialOrd + std::ops::Neg<Output = T> + Copy,
{
    fn abs(&self) -> T {
        let def = T::default();
        match self.partial_cmp(&def) {
            Some(std::cmp::Ordering::Greater) => *self,
            Some(std::cmp::Ordering::Equal) => def,
            Some(std::cmp::Ordering::Less) => -*self,
            None => *self,
        }
    }
}

pub fn pixel_wise_matching_with_tol<T, U, V>(
    matching_array_left: &ArrayView3<T>,
    differencing_array_left: &ArrayView3<U>,
    matching_array_right: &ArrayView3<T>,
    differencing_array_right: &ArrayView3<U>,
    tol_match: V,
    invalid: T,
) -> Result<Triplet<T, U>, ShapeError>
where
    T: std::ops::Sub<T, Output = V> + std::cmp::PartialEq + Copy + Send + Sync,
    U: std::ops::Sub<U, Output = U> + std::cmp::PartialEq + Copy + Send + Sync,
    V: Absolute + std::cmp::PartialOrd + Send + Sync,
{
    if matching_array_left.shape() != differencing_array_left.shape()
        || matching_array_right.shape() != differencing_array_right.shape()
        || matching_array_left.shape()[..2] != matching_array_right.shape()[..2]
    {
        return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
    }

    let mut vals_out: Vec<U> = Vec::default();
    let mut time_left: Vec<T> = Vec::default();
    let mut time_right: Vec<T> = Vec::default();

    Zip::from(matching_array_left.lanes(Axis(2)))
        .and(matching_array_right.lanes(Axis(2)))
        .and(differencing_array_left.lanes(Axis(2)))
        .and(differencing_array_right.lanes(Axis(2)))
        .for_each(|ml, mr, dl, dr| {
            Zip::from(ml).and(dl).for_each(|&mli, &dli| {
                Zip::from(mr).and(dr).for_each(|&mri, &dri| {
                    let dd = (mli - mri).abs();
                    if dd < tol_match && mli != invalid && mri != invalid {
                        vals_out.push(dli - dri);
                        time_right.push(mri);
                        time_left.push(mli);
                    }
                })
            })
        });

    Ok((time_left, time_right, vals_out))
}

pub fn pixel_wise_matching_with_tol_indexes<T, V>(
    matching_array_left: &ArrayView3<T>,
    matching_array_right: &ArrayView3<T>,
    tol_match: V,
    invalid: T,
) -> Result<Quadruplet<usize>, ShapeError>
where
    T: std::ops::Sub<T, Output = V> + std::cmp::PartialEq + Copy + Send + Sync,
    V: Absolute + std::cmp::PartialOrd + Send + Sync,
{
    if matching_array_left.shape()[..2] != matching_array_right.shape()[..2] {
        return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
    }

    let mut indexes_x: Vec<usize> = Vec::default();
    let mut indexes_y: Vec<usize> = Vec::default();

    let mut indexes_left: Vec<usize> = Vec::default();
    let mut indexes_right: Vec<usize> = Vec::default();

    Zip::indexed(matching_array_left.lanes(Axis(2)))
        .and(matching_array_right.lanes(Axis(2)))
        .for_each(|(ix, iy), ml, mr| {
            Zip::indexed(ml).for_each(|i_l, &mli| {
                Zip::indexed(mr).for_each(|i_r, &mri| {
                    let dd = (mli - mri).abs();
                    if dd < tol_match && mli != invalid && mri != invalid {
                        indexes_x.push(ix);
                        indexes_y.push(iy);
                        indexes_left.push(i_l);
                        indexes_right.push(i_r);
                    }
                })
            })
        });

    Ok((indexes_x, indexes_y, indexes_left, indexes_right))
}
