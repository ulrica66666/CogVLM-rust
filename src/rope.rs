use ndarray::{Array2};
// use std::f32::consts::PI;

pub fn apply_rope(tensor: &mut Array2<f32>, dim: usize) {
    let seq_len = tensor.shape()[0];
    let theta: Vec<f32> = (0..dim / 2)
        .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / dim as f32))
        .collect();

    for pos in 0..seq_len {
        for i in 0..(dim / 2) {
            let angle = pos as f32 * theta[i];
            let (sin, cos) = angle.sin_cos();

            let (x0, x1) = (tensor[[pos, 2 * i]], tensor[[pos, 2 * i + 1]]);
            tensor[[pos, 2 * i]] = cos * x0 - sin * x1;
            tensor[[pos, 2 * i + 1]] = sin * x0 + cos * x1;
        }
    }
}
