use {
    crate::{
        att::{Cos, Hemisphere},
        dev::MaterialKind,
        Dtype, Glass, Hitee, Interaction, Light, Material, Ray, Scene, Tracer, Vec3,
    },
    rand::prelude::*,
    rand_distr::{uniform::SampleUniform, Distribution, Standard},
    std::ops::{Mul, MulAssign},
};

#[derive(Default, Debug, Clone, Copy)]
pub struct PathTracer {
    pub depth: usize,
}

impl<F: Dtype + SampleUniform + MulAssign + Mul> Tracer<F> for PathTracer
where
    Standard: Distribution<F>,
{
    fn trace<R: Rng>(&self, mut ray: Ray<F>, scene: &Scene<F>, rng: &mut R) -> Option<Vec3<F>> {
        let t_min = F::from_f64(1e-3f64).unwrap();
        let t_max = F::max_value().unwrap();

        let mut vatt: Vec3<F> = Vec3::from([F::one(); 3]);
        for _ in 0..self.depth {
            match scene.shoot_at(ray, t_min, t_max) {
                Some(hit) => {
                    let material = scene.material_for(hit.obj_idx);
                    match material_interaction(material, ray.dir, hit.normal, rng) {
                        Interaction::Scatter { dir, att } => {
                            vatt.component_mul_assign(&att);
                            ray.ori = hit.pos;
                            ray.dir = dir;
                        }
                        Interaction::Emit { emission } => {
                            return Some(vatt.component_mul(&emission))
                        }
                    }
                }
                None => break,
            }
        }
        None
    }
}

pub(crate) fn material_interaction<F, R>(
    material: &Material<F>,
    in_direction: Vec3<F>,
    normal: Vec3<F>,
    rng: &mut R,
) -> Interaction<F>
where
    R: Rng,
    F: Dtype + SampleUniform + MulAssign,
    Standard: Distribution<F>,
{
    match material.kind {
        MaterialKind::Diffuse => diffuse(material.rgb, normal, rng),
        MaterialKind::Mirror => mirror(material.rgb, in_direction, normal),
        MaterialKind::Glass(Glass { ior }) => {
            // dielectric_interaction(material.rgb, in_direction, normal, rng)
            todo!()
        }
        MaterialKind::Light(Light { hv }) => light(material.rgb, hv),
    }
}

pub(crate) fn diffuse<F, R>(rgb: Vec3<F>, normal: Vec3<F>, rng: &mut R) -> Interaction<F>
where
    R: Rng,
    F: Dtype + SampleUniform + MulAssign,
    Standard: Distribution<F>,
{
    let dist = Cos::normal(normal);
    let dir = dist.sample(rng);
    Interaction::Scatter {
        att: rgb * F::frac_1_pi() * dir.dot(&normal).abs() / dist.att(&dir),
        dir,
    }
}

pub(crate) fn mirror<F: Dtype + SampleUniform + MulAssign>(
    rgb: Vec3<F>,
    dir: Vec3<F>,
    normal: Vec3<F>,
) -> Interaction<F> {
    Interaction::Scatter { att: rgb, dir: reflect(dir, normal) }
}

// pub(crate) fn dielectric_interaction<F, R>(
//     dielectric: &Glass<F>,
//     in_direction: &Three<F>,
//     normal: &Three<F>,
//     rng: &mut R,
// ) -> LightInteraction<F>
// where
//     F: Dtype + ToPrimitive + SampleUniform,
//     Standard: Distribution<F>,
//     R: Rng,
// {
//     let cos_theta = in_direction.dot(normal);
//     let exiting = cos_theta > F::zero();
//     let outward_normal = &if exiting { -*normal } else { *normal };
//     let ratio = if exiting { dielectric.ior } else { dielectric.ior.recip() };
//     let cos_theta = cos_theta.abs();
//     let sin_theta = (F::one() - cos_theta.powi(2)).sqrt();
//
//     let direction = if ratio * sin_theta > F::one() {
//         reflect(in_direction, outward_normal)
//     } else {
//         // shclick approximation
//         let r0 = (F::one() - ratio) / (F::one() + ratio);
//         let r1 = r0 * r0;
//         let reflectance = r1 + (F::one() - r1) * (F::one() - cos_theta).powi(5);
//
//         if reflectance > Standard.sample(rng) {
//             reflect(in_direction, outward_normal)
//         } else {
//             // refract
//             let perp = (in_direction + &(outward_normal * cos_theta)) * ratio;
//             let para = outward_normal * -(F::one() - perp.length_squared()).abs().sqrt();
//             (perp + para).normalized()
//         }
//     };
//
//     LightInteraction::Scatter { attenuation: dielectric.rgb, direction }
// }

pub(crate) fn light<F: Dtype>(rgb: Vec3<F>, hv: F) -> Interaction<F> {
    Interaction::Emit { emission: rgb * hv }
}

fn reflect<F: Dtype>(d: Vec3<F>, n: Vec3<F>) -> Vec3<F> {
    d - &(n * (d.dot(&n) * F::from_f64(2.0f64).unwrap()))
}
