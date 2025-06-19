use image::{DynamicImage, RgbImage, Rgb};
use ndarray::Array3;
use rayon::prelude::*;

const DEFAULT_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const DEFAULT_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

pub struct ImageProcessor {
    pub image_size: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl ImageProcessor {
    pub fn new(image_size: u32) -> Self {
        Self {
            image_size,
            mean: DEFAULT_MEAN,
            std: DEFAULT_STD,
        }
    }

    pub fn preprocess(&self, img: &DynamicImage) -> Array3<f32> {
        let resized = resize_bicubic(&img.to_rgb8(), self.image_size, self.image_size);
        let mut arr = Array3::<f32>::zeros((3, self.image_size as usize, self.image_size as usize));
        for (x, y, pixel) in resized.enumerate_pixels() {
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                arr[[c, y as usize, x as usize]] = (val - self.mean[c]) / self.std[c];
            }
        }
        arr
    }
}

pub struct ImageBatchOutput {
    pub image: Vec<Array3<f32>>,
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
}

pub fn process_images_in_batch(
    images: Vec<DynamicImage>,
    processor: &ImageProcessor,
) -> ImageBatchOutput {
    let processed_images: Vec<_> = images
        .par_iter()
        .map(|img| processor.preprocess(img))
        .collect();

    let batch_size = processed_images.len();
    ImageBatchOutput {
        image: processed_images,
        input_ids: vec![0; batch_size],
        attention_mask: vec![1; batch_size],
    }
}

fn cubic_kernel(x: f32) -> f32 {
    let abs_x = x.abs();
    if abs_x < 1.0 {
        (1.5 * abs_x.powi(3)) - (2.5 * abs_x.powi(2)) + 1.0
    } else if abs_x < 2.0 {
        (-0.5 * abs_x.powi(3)) + (2.5 * abs_x.powi(2)) - (4.0 * abs_x) + 2.0
    } else {
        0.0
    }
}


fn resize_bicubic(input: &RgbImage, out_w: u32, out_h: u32) -> RgbImage {
    let (in_w, in_h) = input.dimensions();
    let mut out = RgbImage::new(out_w, out_h);

    for y in 0..out_h {
        let fy = (y as f32 + 0.5) * (in_h as f32 / out_h as f32) - 0.5;
        let y_int = fy.floor() as i32;
        let y_frac = fy - y_int as f32;

        for x in 0..out_w {
            let fx = (x as f32 + 0.5) * (in_w as f32 / out_w as f32) - 0.5;
            let x_int = fx.floor() as i32;
            let x_frac = fx - x_int as f32;

            let mut rgb = [0.0f32; 3];

            for m in -1..=2 {
                let wy = cubic_kernel(m as f32 - y_frac);
                let sy = y_int + m;
                if sy < 0 || sy >= in_h as i32 {
                    continue;
                }

                for n in -1..=2 {
                    let wx = cubic_kernel(x_frac - n as f32);
                    let sx = x_int + n;
                    if sx < 0 || sx >= in_w as i32 {
                        continue;
                    }

                    let pixel = input.get_pixel(sx as u32, sy as u32);
                    for c in 0..3 {
                        rgb[c] += pixel[c] as f32 * wx * wy;
                    }
                }
            }

            let pixel = image::Rgb([
                rgb[0].clamp(0.0, 255.0) as u8,
                rgb[1].clamp(0.0, 255.0) as u8,
                rgb[2].clamp(0.0, 255.0) as u8,
            ]);
            out.put_pixel(x, y, pixel);
        }
    }

    out
}

