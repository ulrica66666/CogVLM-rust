// src/patch_dropout.rs

use ndarray::{Array2, Axis, concatenate, s};
use rand::seq::index::sample;
use rand::thread_rng;
use rayon::prelude::*;
use std::simd::Simd;  

pub struct PatchDropout {
    pub keep_ratio: f32,
    pub cls_token: bool,
}

impl PatchDropout {
    pub fn new(keep_ratio: f32, cls_token: bool) -> Self {
        PatchDropout { keep_ratio, cls_token }
    }
    // 原始版本
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let (n, _) = x.dim();
        if self.keep_ratio >= 1.0 {
            return x.clone();
        }

        let mut rng = thread_rng();
        let keep_count = ((n as f32) * self.keep_ratio).round() as usize;

        let (opt_cls, patches) = if self.cls_token {
            let cls = x.slice(s![0..1, ..]).to_owned();
            let pats = x.slice(s![1.., ..]).to_owned();
            (Some(cls), pats)
        } else {
            (None, x.clone())
        };

        let patch_count = patches.shape()[0];
        let sample_count = keep_count.saturating_sub(opt_cls.as_ref().map_or(0, |_| 1));
        let indices = sample(&mut rng, patch_count, sample_count).into_vec();

        let mut arrays = Vec::with_capacity(sample_count + opt_cls.is_some() as usize);
        if let Some(cls_arr) = opt_cls {
            arrays.push(cls_arr);
        }
        for idx in indices {
            let row = patches.row(idx).to_owned().insert_axis(Axis(0));
            arrays.push(row);
        }

        let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
        concatenate(Axis(0), &views).unwrap()
    }

    // Rayon
    pub fn forward_rayon(&self, x: &Array2<f32>) -> Array2<f32> {
        let (n, _) = x.dim();
        if self.keep_ratio >= 1.0 {
            return x.clone();
        }

        let mut rng = thread_rng();
        let keep_count = ((n as f32) * self.keep_ratio).round() as usize;

        let (opt_cls, patches) = if self.cls_token {
            let cls = x.slice(s![0..1, ..]).to_owned();
            let pats = x.slice(s![1.., ..]).to_owned();
            (Some(cls), pats)
        } else {
            (None, x.clone())
        };

        let patch_count = patches.shape()[0];
        let sample_count = keep_count.saturating_sub(opt_cls.as_ref().map_or(0, |_| 1));
        let indices = sample(&mut rng, patch_count, sample_count).into_vec();

        let mut arrays: Vec<_> = indices
            .into_par_iter()
            .map(|idx| patches.row(idx).to_owned().insert_axis(Axis(0)))
            .collect();

        if let Some(cls_arr) = opt_cls {
            arrays.insert(0, cls_arr);
        }

        let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
        concatenate(Axis(0), &views).unwrap()
    }

    // simd
    pub fn forward_rayon_simd(&self, x: &Array2<f32>) -> Array2<f32> {
        let (n, d) = x.dim();
        if self.keep_ratio >= 1.0 {
            return x.clone();
        }

        let mut rng = thread_rng();
        let keep_count = ((n as f32) * self.keep_ratio).round() as usize;

        let (opt_cls, patches) = if self.cls_token {
            let cls = x.slice(s![0..1, ..]).to_owned();
            let pats = x.slice(s![1.., ..]).to_owned();
            (Some(cls), pats)
        } else {
            (None, x.clone())
        };

        let patch_count = patches.shape()[0];
        let sample_count = keep_count.saturating_sub(opt_cls.as_ref().map_or(0, |_| 1));
        let indices = sample(&mut rng, patch_count, sample_count).into_vec();

        let mut arrays: Vec<_> = indices
            .into_par_iter()
            .map(|idx| {
                let row = patches.row(idx);
                let buf = copy_row_simd(row.as_slice().unwrap());
                Array2::from_shape_vec((1, d), buf).unwrap()
            })
            .collect();

        if let Some(cls_arr) = opt_cls {
            arrays.insert(0, cls_arr);
        }

        let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
        concatenate(Axis(0), &views).unwrap()
    }
}

// simd加速拷贝 每次处理LANES个float
fn copy_row_simd(input: &[f32]) -> Vec<f32> {
    const LANES: usize = 8;
    type SimdType = Simd<f32, LANES>;

    let mut output = Vec::with_capacity(input.len());
    let chunks = input.chunks_exact(LANES);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let v = SimdType::from_slice(chunk);
        output.extend_from_slice(&v.to_array());
    }
    output.extend_from_slice(remainder);
    output
}