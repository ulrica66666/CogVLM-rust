// examples/test_full_encoder.rs
extern crate cogvlm_image_preprocessor;

use image::{open, GenericImageView};
use ndarray::{s, Array2, Array3};
use cogvlm_image_preprocessor::processor::ImageProcessor;
use cogvlm_image_preprocessor::patch_embed::PatchEmbed;
use cogvlm_image_preprocessor::patch_dropout::PatchDropout;
use cogvlm_image_preprocessor::rope::{apply_rope};
use cogvlm_image_preprocessor::transformer::TransformerLayer;
use cogvlm_image_preprocessor::glu_projection::GLUProjection;

fn main() {
    // 预处理图片
    let img = open("examples/1.jpg").expect("无法加载examples/1.jpg");
    println!("原始图像尺寸: {:?}", img.dimensions());

    let processor = ImageProcessor::new(224);
    let arr3: Array3<f32> = processor.preprocess(&img);
    println!("预处理后Array3形状: {:?}", arr3.dim());
    println!("样本像素值示例: {:?}", arr3.slice(s![..1, ..1, ..1]));

    // PatchEmbed
    let patch_size = 16;
    let embed_dim = 768;
    let embedder = PatchEmbed::new(patch_size, embed_dim);
    let patches: Array2<f32> = embedder.forward(&arr3);
    println!("PatchEmbed输出shape: {:?}", patches.dim());
    println!("第2个patch元素示例: {:?}", patches.row(1).slice(s![..8])); 

    // PatchDropout
    let dropout = PatchDropout::new(0.9, true);
    let dropped = dropout.forward_rayon_simd(&patches);
    println!("PatchDropout.rayon_simd输出shape: {:?}", dropped.dim());
    println!("丢弃后第2个patch元素示例: {:?}", dropped.row(1).slice(s![..8]));

    // rope
    let mut rope_in = dropped.clone();
    apply_rope(&mut rope_in, embed_dim);
    println!("RoPE后第2个patch元素示例: {:?}", rope_in.row(1).slice(s![..8]));

    // Transformer Layer
    let ff_dim = 3072;  
    let num_heads = 12; 

    let block = TransformerLayer::new(embed_dim, ff_dim, num_heads);
    let tr_out: Array2<f32> = block.forward(&rope_in);
    println!("TransformerLayer 输出 shape: {:?}", tr_out.dim());
    println!("Transformer输出第2行元素示例: {:?}", tr_out.row(1).slice(s![..8]));

    // GLU Projection
    let out_dim = 512;
    let glu = GLUProjection::new(embed_dim, out_dim);
    let glu_out = glu.forward_rayon_simd(&tr_out);
    println!("GLUProjection.rayon_simd输出shape: {:?}", glu_out.dim());
    println!("GLU输出第2行元素示例: {:?}", glu_out.row(1).slice(s![..8]));
}
