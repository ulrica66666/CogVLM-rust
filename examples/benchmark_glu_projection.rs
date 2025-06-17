use cogvlm_image_preprocessor::glu_projection::GLUProjection;
use ndarray::Array2;
use std::time::Instant;

fn main() {
    let in_dim = 1024;
    let out_dim = 512;
    let glu = GLUProjection::new(in_dim, out_dim);

    let tensors: Vec<Array2<f32>> = (0..50)
        .map(|_| Array2::ones((256, in_dim)))
        .collect();

    let t0 = Instant::now();
    for x in &tensors {
        let _ = glu.forward(x);
    }
    let orig = t0.elapsed();

    let t1 = Instant::now();
    for x in &tensors {
        let _ = glu.forward_rayon(x);
    }
    let rayon = t1.elapsed();

    let t2 = Instant::now();
    for x in &tensors {
        let _ = glu.forward_rayon_simd(x);
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
