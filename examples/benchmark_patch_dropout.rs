// examples/benchmark_patch_dropout.rs

use cogvlm_image_preprocessor::patch_dropout::PatchDropout;
use ndarray::Array2;
use std::time::Instant;

fn main() {
    // 参数配置
    let keep_ratio = 0.7;
    let cls_token = true;
    let dropout = PatchDropout::new(keep_ratio, cls_token);

    // 构造50个待测输入
    let n = 1024;
    let dim = 1024;
    let tensors: Vec<Array2<f32>> = (0..50)
        .map(|_| Array2::ones((n, dim)))
        .collect();

    // 原始
    let t0 = Instant::now();
    for x in &tensors {
        let _ = dropout.forward(x);
    }
    let orig = t0.elapsed();

    // Rayon
    let t1 = Instant::now();
    for x in &tensors {
        let _ = dropout.forward_rayon(x);
    }
    let rayon = t1.elapsed();

    // Rayon + SIMD
    let t2 = Instant::now();
    for x in &tensors {
        let _ = dropout.forward_rayon_simd(x);
    }
    let simd = t2.elapsed();

    println!("Processed {} tensors:", tensors.len());
    println!("  Original       : {:.2?}", orig);
    println!("  Rayon-only     : {:.2?}", rayon);
    println!("  Rayon + SIMD   : {:.2?}", simd);
    println!("Average per run:");
    println!("  Original       : {:.2?}", orig / tensors.len() as u32);
    println!("  Rayon-only     : {:.2?}", rayon / tensors.len() as u32);
    println!("  Rayon + SIMD   : {:.2?}", simd / tensors.len() as u32);
}
