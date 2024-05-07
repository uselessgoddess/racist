use {
    crate::{Camera, Dtype, Scene, Tracer, Vec3},
    crossbeam::channel,
    image::{Rgb, RgbImage},
    indicatif::{ProgressBar, ProgressStyle},
    num_traits::{cast, Float},
    rand::{prelude::Rng, SeedableRng},
    rand_distr::{uniform::SampleUniform, Distribution, Standard},
    rayon::prelude::*,
    std::ops::AddAssign,
};

pub fn render<T, F, R>(
    tracer: T,
    scene: Scene<F>,
    camera: Camera<F>,
    num_samples: usize,
) -> RgbImage
where
    T: Tracer<F> + Send + Sync + Default + 'static,
    F: Dtype + SampleUniform + Send + Sync + AddAssign + 'static,
    R: Rng + SeedableRng,
    Standard: Distribution<F>,
{
    let num_pixels = camera.width * camera.height;
    let num_rays = num_pixels * num_samples;

    let (sender, receiver) = channel::bounded(1024);

    let t = std::thread::spawn(move || {
        (0..num_rays)
            .into_par_iter()
            .map(|ray_idx| {
                let mut rng = R::seed_from_u64(ray_idx as u64);
                let pixel_idx = ray_idx % num_pixels;
                let y: F = F::from_usize(pixel_idx / camera.width).unwrap();
                let x: F = F::from_usize(pixel_idx % camera.width).unwrap();
                let jx = x + Standard.sample(&mut rng);
                let jy = y + Standard.sample(&mut rng);
                let ray = camera.ray_through(jx, jy);
                let opt_color = tracer.trace(ray, &scene, &mut rng);
                (pixel_idx, opt_color.unwrap_or(Vec3::zeros()))
            })
            .for_each_with(sender, |s, x| s.send(x).unwrap());
    });

    let pb = ProgressBar::new(num_rays as u64).with_style(
        ProgressStyle::default_bar()
            .template("{bar:40} {elapsed_precise}<{eta} {per_sec}")
            .unwrap(),
    );
    // pb.set_draw_rate(1); // NOTE: indicatif drawing is bottleneck with rayon because of high speeds

    let mut colors = camera.blank();
    for (pixel_idx, color) in receiver.iter() {
        colors[pixel_idx] += color;
        pb.inc(1);
    }

    t.join().unwrap();

    let mut img = RgbImage::new(camera.width as u32, camera.height as u32);
    for x in 0..camera.width {
        for y in 0..camera.height {
            let mean_color = colors[y * camera.width + x] / F::from_usize(num_samples).unwrap();
            img.put_pixel(x as u32, y as u32, into_rgb8(mean_color));
        }
    }
    img
}

fn into_rgb8<F: Dtype>(v: Vec3<F>) -> Rgb<u8> {
    let _0 = F::zero();
    let _1 = F::one();
    let _255 = F::from_u8(255).unwrap();
    Rgb([
        (v.x.clamp(_0, _1) * _255).round().to_u8().unwrap(),
        (v.y.clamp(_0, _1) * _255).round().to_u8().unwrap(),
        (v.z.clamp(_0, _1) * _255).round().to_u8().unwrap(),
    ])
}
