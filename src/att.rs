use {
    crate::{Dtype, Vec3},
    rand::Rng,
    rand_distr::{Distribution, Standard},
};

pub trait Hemisphere<F> {
    fn sample<R: Rng>(&self, rng: &mut R) -> Vec3<F>;

    fn att(&self, v: &Vec3<F>) -> F;

    fn sample_att<R: Rng>(&self, rng: &mut R) -> (Vec3<F>, F) {
        let dir = self.sample(rng);
        let att = self.att(&dir);
        (dir, att)
    }
}

pub struct Cos<F> {
    normal: Vec3<F>,
    u: Vec3<F>,
    v: Vec3<F>,
}

impl<F: Dtype> Cos<F> {
    pub fn normal(normal: Vec3<F>) -> Self {
        let a = if normal.x.abs() > F::from_f64(0.9f64).unwrap() {
            Vec3::new(F::zero(), F::one(), F::zero())
        } else {
            Vec3::new(F::one(), F::zero(), F::zero())
        };

        let u = normal.cross(&a).normalize();
        let v = normal.cross(&u).normalize();

        Self { normal, u, v }
    }
}

impl<F> Hemisphere<F> for Cos<F>
where
    F: Dtype,
    Standard: Distribution<F>,
{
    fn sample<R: Rng>(&self, rng: &mut R) -> Vec3<F> {
        // sample local random cosine direction
        let r = Standard.sample(rng);
        let z = (F::one() - r).sqrt();
        let phi = F::from_f64(2.0f64).unwrap() * F::pi() * Standard.sample(rng);
        let y = phi.sin() * r.sqrt();
        let x = phi.cos() * r.sqrt();

        // transform to world coordinates using u/v/normal basis
        &self.u * x + &self.v * y + &self.normal * z
    }

    fn att(&self, v: &Vec3<F>) -> F {
        v.dot(&self.normal).abs() * F::frac_1_pi()
    }
}
