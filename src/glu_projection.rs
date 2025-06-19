use ndarray::{Array2, Axis, s};
use rayon::prelude::*;
use std::simd::{Simd};
use std::simd::num::SimdFloat;
use ndarray_rand::RandomExt;
use rand_distr::Uniform;

pub struct GLUProjection {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weight: Array2<f32>,      // [in_dim, 2*out_dim]
    pub bias: Option<Array2<f32>> // [1, 2*out_dim]
}

impl GLUProjection {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let fan_in = in_dim;
        let fan_out = 2 * out_dim;
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
        let dist = Uniform::new(-limit, limit);

        let weight = Array2::random((in_dim, 2 * out_dim), dist);
        let bias = Some(Array2::zeros((1, 2 * out_dim)));

        GLUProjection { in_dim, out_dim, weight, bias }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut projected = x.dot(&self.weight);
        if let Some(bias) = &self.bias {
            projected += bias;
        }

        let value_part = projected.slice(s![.., 0..self.out_dim]).to_owned();
        let gate_part = projected.slice(s![.., self.out_dim..]).mapv(|v| v.max(0.0));

        value_part * gate_part
    }

    pub fn forward_rayon(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut projected = x.dot(&self.weight);
        if let Some(bias) = &self.bias {
            projected += bias;
        }

        let rows: Vec<_> = projected.axis_iter(Axis(0)).map(|r| r.to_owned()).collect();
        let activated_rows: Vec<_> = rows
            .into_par_iter()
            .map(|row| activate_glu(row, self.out_dim))
            .collect();

        ndarray::stack(Axis(0), &activated_rows.iter().map(|r| r.view()).collect::<Vec<_>>()).unwrap()
    }

    pub fn forward_rayon_simd(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut projected = x.dot(&self.weight);
        if let Some(bias) = &self.bias {
            projected += bias;
        }

        let rows: Vec<_> = projected.axis_iter(Axis(0)).map(|r| r.to_owned()).collect();
        let output_rows: Vec<_> = rows
            .into_par_iter()
            .map(|row| activate_glu_simd(row, self.out_dim))
            .collect();

        ndarray::stack(Axis(0), &output_rows.iter().map(|r| r.view()).collect::<Vec<_>>()).unwrap()
    }
}

// 标准 GLU 激活函数
fn activate_glu(row: ndarray::Array1<f32>, out_dim: usize) -> ndarray::Array1<f32> {
    let value = row.slice(s![0..out_dim]).to_owned();
    let gate = row.slice(s![out_dim..]).mapv(|v| v.max(0.0)); // ReLU 激活门控
    value * gate
}

// SIMD
fn activate_glu_simd(row: ndarray::Array1<f32>, out_dim: usize) -> ndarray::Array1<f32> {
    const LANES: usize = 8;
    type SimdType = Simd<f32, LANES>;

    let value = row.slice(s![0..out_dim]);
    let gate = row.slice(s![out_dim..]);

    let mut out = Vec::with_capacity(out_dim);

    let value_chunks = value.as_slice().unwrap().chunks_exact(LANES);
    let gate_chunks = gate.as_slice().unwrap().chunks_exact(LANES);
    let remainder_v = value_chunks.remainder();
    let remainder_g = gate_chunks.remainder();

    for (vc, gc) in value_chunks.zip(gate_chunks) {
        let v_simd = SimdType::from_slice(vc);
        let g_simd = SimdType::from_slice(gc);
        let relu_gate = g_simd.simd_max(SimdType::splat(0.0));
        let out_simd = v_simd * relu_gate;
        out.extend_from_slice(&out_simd.to_array());
    }

    for i in 0..remainder_v.len() {
        let v = remainder_v[i];
        let g = remainder_g[i].max(0.0);
        out.push(v * g);
    }

    ndarray::Array1::from(out)
}
