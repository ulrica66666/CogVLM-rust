use image::{open, ImageOutputFormat};
use cogvlm_image_preprocessor::processor::{resize_bicubic};
use std::fs::File;
use std::path::Path;

fn main() {
    // 设置路径
    let input_path = "examples/1.jpg";
    let output_path = "rust_resized.png";
    let target_size = 224;

    // 加载图像
    let img = open(input_path)
        .expect("无法打开输入图像")
        .to_rgb8();

    // 使用 resize_bicubic 函数
    let resized = resize_bicubic(&img, target_size, target_size);

    // 保存图像
    let path = Path::new(output_path);
    let file = File::create(path).expect("无法创建输出文件");
    resized
        .write_to(&mut std::io::BufWriter::new(file), ImageOutputFormat::Png)
        .expect("无法写入图像文件");

    println!("Resize 完成，保存为 {}", output_path);
}
