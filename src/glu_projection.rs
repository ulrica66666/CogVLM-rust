// src/glu_projection.rs

use ndarray::{Array2, s};

pub struct GLUProjection {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weight: Array2<f32>,
    pub bias: Option<Array2<f32>>,
}

impl GLUProjection {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let weight = Array2::<f32>::zeros((in_dim, 2 * out_dim));
        let bias = Some(Array2::<f32>::zeros((1, 2 * out_dim)));
        GLUProjection { in_dim, out_dim, weight, bias }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut projected = x.dot(&self.weight);
        if let Some(bias) = &self.bias {
            projected = &projected + bias;
        }

        // 拆分 value 和 gate 两个部分
        let value_part = projected.slice(s![.., 0..self.out_dim]).to_owned();
        let gate_part = projected
            .slice(s![.., self.out_dim..])
            .mapv(|v| v.max(0.0));  // 这里用 ReLU 作为例子

        value_part * gate_part
    }
}
