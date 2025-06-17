use ndarray::{Array3, Array2, Axis, s};
use rayon::prelude::*;
use wide::f32x4;

// rayon+simd优化
pub fn apply_rope_simd_parallel(tensor: &mut Array2<f32>, dim: usize) {
    use wide::f32x4;

    let seq_len = tensor.shape()[0];

    let theta: Vec<f32> = (0..dim / 2)
        .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / dim as f32))
        .collect();

    let rows: Vec<_> = tensor.axis_iter_mut(Axis(0)).collect();

    rows.into_par_iter().enumerate().for_each(|(pos, mut row)| {
        let pos_f = pos as f32;

        let mut i = 0;
        while i + 1 < dim / 2 {
            // 取两个旋转对
            let angle0 = pos_f * theta[i];
            let angle1 = pos_f * theta[i + 1];

            let (sin0, cos0) = angle0.sin_cos();
            let (sin1, cos1) = angle1.sin_cos();

            // 构造simd向量
            // 角度向量格式: [cos0, cos1, sin0, sin1]
            let cos_vec = f32x4::from([cos0, cos1, cos0, cos1]);
            let sin_vec = f32x4::from([sin0, sin1, sin0, sin1]);

            let x0_0 = row[2 * i];
            let x1_0 = row[2 * i + 1];
            let x0_1 = row[2 * (i + 1)];
            let x1_1 = row[2 * (i + 1) + 1];

            let x_vec = f32x4::from([x0_0, x0_1, x1_0, x1_1]);

            let x0_vec = f32x4::from([x0_0, x0_1, 0.0, 0.0]);
            let x1_vec = f32x4::from([x1_0, x1_1, 0.0, 0.0]);

            // 旋转计算：
            // 对第一个旋转对
            let new_x0_0 = cos0 * x0_0 - sin0 * x1_0;
            let new_x1_0 = sin0 * x0_0 + cos0 * x1_0;
            // 对第二个旋转对
            let new_x0_1 = cos1 * x0_1 - sin1 * x1_1;
            let new_x1_1 = sin1 * x0_1 + cos1 * x1_1;

            // 写回
            row[2 * i] = new_x0_0;
            row[2 * i + 1] = new_x1_0;
            row[2 * (i + 1)] = new_x0_1;
            row[2 * (i + 1) + 1] = new_x1_1;

            i += 2;
        }

        while i < dim / 2 {
            let angle = pos_f * theta[i];
            let (sin, cos) = angle.sin_cos();

            let x0 = row[2 * i];
            let x1 = row[2 * i + 1];
            row[2 * i] = cos * x0 - sin * x1;
            row[2 * i + 1] = sin * x0 + cos * x1;
            i += 1;
        }
    });
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
            tensor[[pos, 2 * i]] = cos * x0 - sin * x1;
            tensor[[pos, 2 * i + 1]] = sin * x0 + cos * x1;
        }
    }
}

// rayon优化
pub fn apply_rope_parallel(tensor: &mut Array2<f32>, dim: usize) {
    let seq_len = tensor.shape()[0];
    let theta: Vec<f32> = (0..dim / 2)
        .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / dim as f32))
        .collect();

    let rows: Vec<_> = tensor.axis_iter_mut(Axis(0)).collect(); // get mutable rows

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