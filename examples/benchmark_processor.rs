use cogvlm_image_preprocessor::processor::{ImageProcessor, process_images_in_batch};
use image::open;
use std::time::Instant;

fn main() {
    let processor = ImageProcessor::new(384);
    let image_paths: Vec<String> = (1..51)
        .map(|i| format!("examples/{}.jpg", i))
        .collect();

    let images = image_paths
        .iter()
        .map(|path| open(path).expect("Failed to open image"))
        .collect::<Vec<_>>();

    let start = Instant::now();
    let result = process_images_in_batch(images, &processor);
    let elapsed = start.elapsed();

    println!("Processed {} images in {:.2?}", result.image.len(), elapsed);
    println!("Average per image: {:.2?}", elapsed / result.image.len() as u32);
}
