use ndarray::prelude::*;
use ndarray::Zip;
use ordered_float::OrderedFloat;


const BAD_VALUE: f64 = f64::INFINITY;
const BAD_INDEX: i32 = -9999;

fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    unsafe {
        indices.sort_by_key(|&i| data.get_unchecked(i));
    }
    indices
}

pub fn find_topk_with_tolerance(
        left: ArrayView1<f64>,
        right: ArrayView1<f64>,
        tolerance: f64,
        topk: usize,
    ) -> Array2<i32> {
        let mut out = Array2::default((left.len(), topk));

        Zip::from(left)
        .and(out.rows_mut())
        .par_for_each(|v_l, mut row_out| {
            let dist = (&right - *v_l).mapv(f64::abs);
            let dist = dist.mapv(|elem| if elem > tolerance {OrderedFloat::from(BAD_VALUE)} else {OrderedFloat::from(elem)});
            let res = argsort(&dist.to_vec());

            let mut valid_row: Vec<i32> = Vec::with_capacity(res.len());
            for i in res.iter() {
                if dist.get(*i).unwrap() > &OrderedFloat::from(tolerance) {
                    valid_row.push(BAD_INDEX);
                } else { valid_row.push(*i as i32)}
            }

            assert!(topk < valid_row.len());
            unsafe { row_out.assign(&Array::from_vec(valid_row.get_unchecked(0..topk).to_vec())) }
        });
        out
    }


#[cfg(test)]
mod matching_tests {
    use super::*;

    #[test]
    fn find_topk_with_tolerance_test() {
        let v1 = Array1::<f64>::from_vec(vec![
            100.125,
            6.5489,
            6.5488,
            0.00,
            -19.0,
            77777.777]);
        println!(" {}", v1);
        let v2 = Array1::<f64>::from_vec(vec![
            0.5,
            100.01,
            99.01,
            6.5488,
            7.0,
            77778.777,
            10.0,
            -20.0,
            3.24,
            ]);
        println!(" {}", v2);
        let test = find_topk_with_tolerance(v1.view(), v2.view(), 5.0, 2);
        assert_eq!(test[[0, 0]], 1)
    }
}

    


