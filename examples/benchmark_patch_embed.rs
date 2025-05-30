use cogvlm_image_preprocessor::patch_embed::PatchEmbed;
use image::open;
use std::time::Instant;
use ndarray::Array3;

fn main() {
    let patch_size = 16;
    let embed_dim = 1024;
    let patch_embed = PatchEmbed::new(patch_size, embed_dim);

    let image_paths: Vec<String> = (1..51)
        .map(|i| format!("examples/{}.jpg", i))
        .collect();

    let mut tensors = vec![];

    for path in image_paths {
        let img = open(&path).expect("Failed to open image").to_rgb8();
        let (w, h) = img.dimensions();
        let array: Array3<f32> = Array3::from_shape_fn((3, h as usize, w as usize), |(c, y, x)| {
            img.get_pixel(x as u32, y as u32)[c] as f32 / 255.0
        });
        tensors.push(array);
    }

    let start = Instant::now();
    for tensor in &tensors {
        let _ = patch_embed.forward(tensor);
    }
    let elapsed = start.elapsed();

    println!("Processed {} images in {:.2?}", tensors.len(), elapsed);
    println!("Average per image: {:.2?}", elapsed / tensors.len() as u32);
}
