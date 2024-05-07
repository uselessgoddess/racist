use {
    crate::{Dtype, Hit, Hitee, Ray, Surface, Vec3},
    num_traits::Float,
    rand::{distributions::Uniform, Rng},
    rand_distr::{uniform::SampleUniform, Distribution, UnitSphere},
};

#[derive(Debug, Clone)]
pub struct Sphere<F> {
    pub(super) center: Vec3<F>,
    pub(super) radius_squared: F,
}

impl<F: Dtype> Sphere<F> {
    pub fn new(x: F, y: F, z: F) -> Self {
        Self { center: Vec3::new(x, y, z), radius_squared: F::one() }
    }

    pub fn scaled(&self, scalar: F) -> Self {
        Self { center: self.center, radius_squared: self.radius_squared * scalar.powi(2) }
    }
}

impl<F: Dtype> Hitee<F> for Sphere<F> {
    fn shoot_at(&self, ray: Ray<F>, t_min: F, t_max: F) -> Option<Hit<F>> {
        let rv = ray.ori - self.center;
        let a = F::one();
        let half_b = rv.dot(&ray.dir);
        let c = rv.norm_squared() - self.radius_squared;

        let discriminant = half_b.powi(2) - a * c;
        if discriminant < F::zero() {
            return None;
        }
        let sqrtd = discriminant.sqrt();

        let near_root = Some((-half_b - sqrtd) * a.recip()).filter(|&v| t_min <= v && v < t_max);
        let far_root = Some((-half_b + sqrtd) * a.recip()).filter(|&v| t_min <= v && v < t_max);
        near_root.or(far_root).map(|len| {
            let offset = ray.dir * len;
            let pos = ray.ori + &offset;
            let normal = (pos - self.center).normalize();
            Hit { pos, len, normal, obj_idx: 0 }
        })
    }
}

fn sample_sphere<F: Dtype + SampleUniform, R: Rng + ?Sized>(rng: &mut R) -> [F; 3] {
    let uniform = Uniform::new(F::from_f64(-1.).unwrap(), F::from_f64(1.).unwrap());
    loop {
        let (x1, x2) = (uniform.sample(rng), uniform.sample(rng));
        let sum = x1 * x1 + x2 * x2;
        if sum >= F::from_f64(1.).unwrap() {
            continue;
        }
        let factor = F::from_f64(2.).unwrap() * (F::one() - sum).sqrt();
        return [
            x1 * factor,
            x2 * factor,
            F::from_f64(1.).unwrap() - F::from_f64(2.).unwrap() * sum,
        ];
    }
}

impl<F: Dtype + SampleUniform> Surface<F> for Sphere<F> {
    fn surface_point<R: Rng>(&self, rng: &mut R) -> Vec3<F> {
        let dir = Vec3::from(sample_sphere(rng));
        let len = self.radius_squared.sqrt();
        self.center + dir * len
    }

    fn normal_at(&self, point: Vec3<F>) -> Vec3<F> {
        (point - self.center).normalize()
    }
}
