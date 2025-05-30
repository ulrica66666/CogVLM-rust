// src/patch_dropout.rs

use ndarray::{Array2, Axis, concatenate, s};
use rand::seq::index::sample;
use rand::thread_rng;

pub struct PatchDropout {
    pub keep_ratio: f32,
    pub cls_token: bool,
}

impl PatchDropout {
    pub fn new(keep_ratio: f32, cls_token: bool) -> Self {
        PatchDropout { keep_ratio, cls_token }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let (n, _) = x.dim();
        if self.keep_ratio >= 1.0 {
            return x.clone();
        }

        let mut rng = thread_rng();
        let keep_count = ((n as f32) * self.keep_ratio).round() as usize;

        // 切分 cls_token 与其余 patches
        let (opt_cls, patches) = if self.cls_token {
            let cls = x.slice(s![0..1, ..]).to_owned();
            let pats = x.slice(s![1.., ..]).to_owned();
            (Some(cls), pats)
        } else {
            (None, x.clone())
        };

        let patch_count = patches.shape()[0];
        // 实际保留的 patches 数量
        let sample_count = keep_count.saturating_sub(opt_cls.as_ref().map_or(0, |_| 1));

        // 随机取 sample_count 个索引
        let indices = sample(&mut rng, patch_count, sample_count).into_vec();

        // 构造新的 token 列表
        let mut arrays: Vec<Array2<f32>> = Vec::with_capacity(sample_count + opt_cls.is_some() as usize);
        if let Some(cls_arr) = opt_cls {
            arrays.push(cls_arr);
        }
        for idx in indices {
            // 按行提取 patches[row(idx), ..]
            let row = patches.row(idx).to_owned().insert_axis(Axis(0));
            arrays.push(row);
        }

        // 把所有 Array2 视图都转成视图切片，然后 stack
        let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
        concatenate(Axis(0), &views).unwrap()
    }
}
