use ndarray::{Array2, Array3, s};
use rayon::prelude::*;
use std::simd::{Simd};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;

pub struct PatchEmbed {
    pub patch_size: usize,
    pub embed_dim: usize,
    pub weight: Array2<f32>, // shape: [embed_dim, patch_dim]
    pub bias: Option<Array2<f32>>, // shape: [embed_dim, 1]
}

impl PatchEmbed {
    pub fn new(patch_size: usize, embed_dim: usize) -> Self {
        let patch_dim = patch_size * patch_size * 3;
        // let weight = Array2::<f32>::zeros((embed_dim, patch_dim));
        // let bias = None;
        let weight = Array2::random((embed_dim, patch_dim), StandardNormal);
        let bias   = Some(Array2::random((embed_dim, 1), StandardNormal));
        PatchEmbed { patch_size, embed_dim, weight, bias }
    }

    pub fn forward(&self, img: &Array3<f32>) -> Array2<f32> {
        let (_, h, w) = (img.shape()[0], img.shape()[1], img.shape()[2]);
        let ph = h / self.patch_size;
        let pw = w / self.patch_size;
        let patch_dim = self.patch_size * self.patch_size * 3;

        let patches: Vec<f32> = (0..ph * pw).into_par_iter()
            .flat_map_iter(|idx| {
                let i = idx / pw;
                let j = idx % pw;
                let patch = img.slice(s![
                    ..,
                    i * self.patch_size..(i + 1) * self.patch_size,
                    j * self.patch_size..(j + 1) * self.patch_size
                ]);

                // 使用 SIMD 加速 flatten
                flatten_patch_simd(&patch.to_owned().into_raw_vec())

            })
            .collect();

        let input = Array2::from_shape_vec((ph * pw, patch_dim), patches).unwrap();
        let mut output = input.dot(&self.weight.t());
        if let Some(bias) = &self.bias {
            output += &bias.t();
        }
        output
    }
}

/// SIMD 加速 flatten patch (向量复制)
fn flatten_patch_simd(input: &[f32]) -> Vec<f32> {
    const LANES: usize = 8;
    type SimdType = Simd<f32, LANES>;

    let mut output = Vec::with_capacity(input.len());
    let chunks = input.chunks_exact(LANES);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let vec = SimdType::from_slice(chunk);
        output.extend_from_slice(&vec.to_array());
    }

    // 剩余数据
    output.extend_from_slice(remainder);
    output
}
