use cogvlm_image_preprocessor::rope::{apply_rope, apply_rope_parallel, apply_rope_simd_parallel};
use ndarray::Array2;
use rand::Rng;
use std::time::Instant;

fn generate_tensor(seq_len: usize, dim: usize) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((seq_len, dim), |_| rng.gen_range(-1.0..1.0))
}

fn main() {
    let seq_len = 1024;
    let dim = 128;
    let n_iters = 1024;

    let mut tensors: Vec<Array2<f32>> = (0..n_iters)
        .map(|_| generate_tensor(seq_len, dim))
        .collect();

    {
        let mut inputs = tensors.clone();
        let start = Instant::now();
        for tensor in &mut inputs {
            apply_rope(tensor, dim);
        }
        let elapsed = start.elapsed();
        println!("apply_rope: processed {} tensors in {:.2?}", n_iters, elapsed);
        println!("Average per tensor: {:.2?}", elapsed / n_iters as u32);
    }

    {
        let mut inputs = tensors.clone();
        let start = Instant::now();
        for tensor in &mut inputs {
            apply_rope_parallel(tensor, dim);
        }
        let elapsed = start.elapsed();
        println!("apply_rope_parallel: processed {} tensors in {:.2?}", n_iters, elapsed);
        println!("Average per tensor: {:.2?}", elapsed / n_iters as u32);
    }

    {
        let mut inputs = tensors.clone();
        let start = Instant::now();
        for tensor in &mut inputs {
            apply_rope_simd_parallel(tensor, dim);
        }
        let elapsed = start.elapsed();
        println!("apply_rope_simd_parallel: processed {} tensors in {:.2?}", n_iters, elapsed);
        println!("Average per tensor: {:.2?}", elapsed / n_iters as u32);
    }
}
