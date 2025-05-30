// examples/test_full_encoder.rs

use cogvlm_image_preprocessor::{
    processor::ImageProcessor,
    patch_embed::PatchEmbed,
    rope::apply_rope,
    transformer::TransformerBlock,
    patch_dropout::PatchDropout,
    glu_projection::GLUProjection,
};
use image::open;
use ndarray::{Array3, Array2, Array1, Axis, stack};
use std::path::Path;

// Helper: 把 Array2 扩展出 cls_token（全零向量）并拼到最前面
fn prepend_cls_token(x: &Array2<f32>) -> Array2<f32> {
    let batch = x.shape()[0];
    let dim = x.shape()[1];
    // 一个全零向量
    let cls = Array2::<f32>::zeros((1, x.shape()[1]));
    // 拼接
    ndarray::concatenate(Axis(0), &[cls.view(), x.view()]).unwrap()
}

fn main() {
    // 1. 读取并预处理图片
    let img = open(Path::new("examples/1.jpg")).expect("Cannot open 1.jpg");
    let processor = ImageProcessor::new(224);
    let img_arr: Array3<f32> = processor.preprocess(&img);
    println!("Preprocessed image shape: {:?}", img_arr.dim());

    // 2. Patch Embed: 切块 + 卷积投影
    let patch_size = 14;
    let embed_dim = 1024; // 以 EVA-CLIP L 版本为例
    let mut patch_embed = PatchEmbed::new(patch_size as usize, embed_dim);
    let mut tokens: Array2<f32> = patch_embed.forward(&img_arr);
    println!("After PatchEmbed: {:?}", tokens.dim());
    //    -> (num_patches, embed_dim), e.g. ( (224/14)^2 = 256, 1024 )

    // 3. 添加 CLS token
    tokens = prepend_cls_token(&tokens);
    println!("After CLS token: {:?}", tokens.dim());
    //    -> (257, 1024)

    // 4. 位置编码（RoPE）
    apply_rope(&mut tokens, embed_dim);
    println!("After RoPE applied.");

    // 5. Transformer Block 堆叠
    let mut x = tokens.clone();
    let num_layers = 4; // 跑 4 层
    let mut blocks: Vec<TransformerBlock> = (0..num_layers)
        .map(|_| TransformerBlock::new(embed_dim))
        .collect();
    for (i, blk) in blocks.iter().enumerate() {
        x = blk.forward(&x);
        println!("After Transformer layer {}: {:?}", i + 1, x.dim());
    }

    // 6. PatchDropout（训练模式下）
    let dropout = PatchDropout::new(0.9, true);
    let x_dropout = dropout.forward(&x);
    println!("After PatchDropout: kept {} tokens", x_dropout.shape()[0]);

    // 7. GLU 投影
    let glu = GLUProjection::new(embed_dim, embed_dim);
    let x_glu = glu.forward(&x_dropout);
    println!("After GLUProjection: {:?}", x_glu.dim());
    println!("Full Vision Encoder output computed successfully.");
}