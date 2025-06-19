use ndarray::{Array2, Axis, s};
use rayon::prelude::*;
use std::simd::{Simd};

// simd
pub fn apply_rope_simd_parallel(tensor: &mut Array2<f32>, dim: usize) {
    const LANES: usize = 8;
    type Vf32 = Simd<f32, LANES>;

    let seq_len = tensor.shape()[0];
    let half_dim = dim / 2;

    let theta: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / dim as f32))
        .collect();

    for pos in 0..seq_len {
        let pos_f32 = pos as f32;
        let mut row = tensor.slice_mut(s![pos, ..]);

        let mut i = 0;
        while i + LANES <= half_dim {
            let angles: [f32; LANES] = {
                let mut arr = [0.0; LANES];
                for lane in 0..LANES {
                    arr[lane] = pos_f32 * theta[i + lane];
                }
                arr
            };

            // 逐元素计算 sin_cos 收集结果到数组
            // TODO:优化
            let mut sin_arr = [0.0; LANES];
            let mut cos_arr = [0.0; LANES];
            for lane in 0..LANES {
                let (s, c) = angles[lane].sin_cos();
                sin_arr[lane] = s;
                cos_arr[lane] = c;
            }

            let sin = Vf32::from_array(sin_arr);
            let cos = Vf32::from_array(cos_arr);

            let mut x0_vals = [0.0; LANES];
            let mut x1_vals = [0.0; LANES];
            for lane in 0..LANES {
                x0_vals[lane] = row[2 * (i + lane)];
                x1_vals[lane] = row[2 * (i + lane) + 1];
            }

            let x0 = Vf32::from_array(x0_vals);
            let x1 = Vf32::from_array(x1_vals);

            let new_x0 = cos * x0 - sin * x1;
            let new_x1 = sin * x0 + cos * x1;

            for lane in 0..LANES {
                row[2 * (i + lane)] = new_x0[lane];
                row[2 * (i + lane) + 1] = new_x1[lane];
            }

            i += LANES;
        }

        while i < half_dim {
            let angle = pos_f32 * theta[i];
            let (sin, cos) = angle.sin_cos();
            let (x0, x1) = (row[2 * i], row[2 * i + 1]);
            row[2 * i] = cos * x0 - sin * x1;
            row[2 * i + 1] = sin * x0 + cos * x1;
            i += 1;
        }
    }
}

// 原版
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
            let new_x0 = cos * x0 - sin * x1;
            let new_x1 = sin * x0 + cos * x1;
            tensor[[pos, 2 * i]] = new_x0;
            tensor[[pos, 2 * i + 1]] = new_x1;
        }
    }
}

// rayon优化
pub fn apply_rope_parallel(tensor: &mut Array2<f32>, dim: usize) {
    let seq_len = tensor.shape()[0];
    let theta: Vec<f32> = (0..dim / 2)
        .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / dim as f32))
        .collect();

    let rows: Vec<_> = tensor.axis_iter_mut(Axis(0)).collect();

    rows.into_par_iter().enumerate().for_each(|(pos, mut row)| {
        for i in 0..(dim / 2) {
            let angle = pos as f32 * theta[i];
            let (sin, cos) = angle.sin_cos();

            let (x0, x1) = (row[2 * i], row[2 * i + 1]);
            row[2 * i] = cos * x0 - sin * x1;
            row[2 * i + 1] = sin * x0 + cos * x1;
        }
    });
}