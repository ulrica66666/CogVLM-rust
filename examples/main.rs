use cogvlm_image_preprocessor::processor::{ImageProcessor, process_images_in_batch};
use ndarray::s;
use image::open;

fn main() {
    let processor = ImageProcessor::new(384);
    let image_paths = vec!["examples/1.jpg"];

    let images = image_paths
        .iter()
        .map(|path| open(path).expect("Failed to open image"))
        .collect::<Vec<_>>();

    let result = process_images_in_batch(images, &processor);

    println!("Processed {} images.", result.image.len());
    
    // 打印第 0 张图像（3通道、384×384）的前 3×3 像素值
    let img_tensor = &result.image[0];
    
    println!("Sample of processed image tensor (C × H × W):");
    for c in 0..3 {
        println!("Channel {}:", c);
        println!("{:?}", img_tensor.slice(s![c, 0..3, 0..3]));
    }
}
