
use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn rdist(a: Vec<f64>, b: Vec<f64>, ncols_a: usize, ncols_b: usize) -> PyResult<Vec<f64>> {
    let mut v: Vec<f64> = Vec::new();

    let mut i = 0;
    while i <= a.len() - ncols_a {
        let x = &a[i..i+ncols_a];

        let mut j = 0;
        while j <= b.len() - ncols_b {
            let y = &b[j..j+ncols_b];
            v.push(x.iter().zip(y).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt());
            j += ncols_b;

        }
        i += ncols_a;
    }
    Ok(v)
}


/// A Python module implemented in Rust.
#[pymodule]
fn pillars(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(rdist, m)?)?;
    Ok(())
}