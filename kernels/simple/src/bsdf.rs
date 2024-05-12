#[allow(dead_code)]
use spirv_std::num_traits::{Float, FloatConst};
use {
    crate::{rng::RngState, util},
    shared::{MaterialData, Sampler, TracingConfig},
    spirv_std::{
        glam::{Vec2, Vec3, Vec4Swizzles},
        Image,
    },
};

#[derive(Default, Copy, Clone, PartialEq)]
#[repr(u32)]
pub enum Lobe {
    #[default]
    DiffuseReflection,
    SpecularReflection,
    DiffuseTransmission,
    SpecularTransmission,
}

type Spectrum = Vec3;

#[derive(Default, Copy, Clone)]
pub struct BSDFSample {
    pub pdf: f32,
    pub lobe: Lobe,
    pub spectrum: Spectrum,
    pub direction: Vec3,
}

pub trait BSDF {
    fn evaluate(&self, view: Vec3, normal: Vec3, sample: Vec3, lobe: Lobe) -> Spectrum;
    fn pdf(&self, view: Vec3, normal: Vec3, sample: Vec3, lobe: Lobe) -> f32;

    fn sample(&self, view: Vec3, normal: Vec3, rng: &mut RngState) -> BSDFSample;
}

pub struct Lambertian {
    pub albedo: Spectrum,
}

impl Lambertian {
    fn pdf_fast(&self, cos_theta: f32) -> f32 {
        cos_theta / f32::PI()
    }

    fn evaluate_fast(&self, cos_theta: f32) -> Spectrum {
        self.albedo / f32::PI() * cos_theta
    }
}

impl BSDF for Lambertian {
    fn evaluate(&self, _view: Vec3, normal: Vec3, sample: Vec3, _lobe: Lobe) -> Spectrum {
        self.evaluate_fast(normal.dot(sample).max(0.0))
    }

    fn pdf(&self, _view: Vec3, normal: Vec3, sample: Vec3, _lobe: Lobe) -> f32 {
        self.pdf_fast(normal.dot(sample).max(0.0))
    }

    fn sample(&self, _view: Vec3, normal: Vec3, rng: &mut RngState) -> BSDFSample {
        let (up, nt, nb) = util::create_cartesian(normal);
        let rng_sample = rng.gen_r3();
        let sample = util::cos_hemisphere(rng_sample.x, rng_sample.y);
        let direction = Vec3::new(
            sample.x * nb.x + sample.y * up.x + sample.z * nt.x,
            sample.x * nb.y + sample.y * up.y + sample.z * nt.y,
            sample.x * nb.z + sample.y * up.z + sample.z * nt.z,
        )
        .normalize();

        let cos_theta = normal.dot(direction).max(0.0);
        let pdf = self.pdf_fast(cos_theta);
        let spectrum = self.evaluate_fast(cos_theta);
        BSDFSample { pdf, lobe: Lobe::DiffuseReflection, spectrum, direction }
    }
}

pub struct Glass {
    pub albedo: Spectrum,
    pub ior: f32,
    pub roughness: f32,
}

impl BSDF for Glass {
    fn evaluate(&self, _view: Vec3, _normal: Vec3, _sample: Vec3, lobe: Lobe) -> Spectrum {
        if lobe == Lobe::SpecularReflection {
            Vec3::ONE
        } else {
            self.albedo
        }
    }

    fn pdf(&self, _view: Vec3, _normal: Vec3, _sample: Vec3, _lobe: Lobe) -> f32 {
        1.0
    }

    fn sample(&self, view: Vec3, normal: Vec3, rng: &mut RngState) -> BSDFSample {
        fn sign(x: f32) -> f32 {
            if x >= 0.0 {
                1.0
            } else {
                -1.0
            }
        }

        let rng_sample = rng.gen_r3();

        let inside = normal.dot(view) < 0.0;
        let normal = if inside { -normal } else { normal };
        let in_ior = if inside { self.ior } else { 1.0 };
        let out_ior = if inside { 1.0 } else { self.ior };

        let microsurface_normal = util::sample_ggx_microsurface_normal(
            rng_sample.x,
            rng_sample.y,
            normal,
            self.roughness,
        );
        let fresnel =
            util::fresnel_schlick_scalar(in_ior, out_ior, microsurface_normal.dot(view).max(0.0));
        if rng_sample.z <= fresnel {
            // Reflection
            let direction = (2.0 * view.dot(microsurface_normal).abs() * microsurface_normal
                - view)
                .normalize();
            let pdf = 1.0;
            let lobe = Lobe::SpecularReflection;
            let spectrum = Vec3::ONE;
            BSDFSample { pdf, lobe, spectrum, direction }
        } else {
            // Refraction
            let eta = in_ior / out_ior;
            let c = view.dot(microsurface_normal);
            let direction = ((eta * c
                - sign(view.dot(normal)) * (1.0 + eta * (c * c - 1.0)).max(0.0).sqrt())
                * microsurface_normal
                - eta * view)
                .normalize();
            let pdf = 1.0;
            let lobe = Lobe::SpecularTransmission;
            let spectrum = self.albedo;
            BSDFSample { pdf, lobe, spectrum, direction }
        }
    }
}

// Assume IOR of 1.5 for dielectrics, which works well for most.
const DIELECTRIC_IOR: f32 = 1.5;

// Fresnel at normal incidence for dielectrics, with air as the other medium.
const DIELECTRIC_F0_SQRT: f32 = (DIELECTRIC_IOR - 1.0) / (DIELECTRIC_IOR + 1.0);
const DIELECTRIC_F0: f32 = DIELECTRIC_F0_SQRT * DIELECTRIC_F0_SQRT;

pub struct PBR {
    pub albedo: Spectrum,
    pub roughness: f32,
    pub metallic: f32,
    pub clamp_weight: Vec2,
}

impl PBR {
    fn evaluate_diffuse_fast(&self, cos_theta: f32, specular_weight: f32, ks: Vec3) -> Spectrum {
        let kd = (Vec3::splat(1.0) - ks) * (1.0 - self.metallic);
        let diffuse = kd * self.albedo / f32::PI();
        diffuse * cos_theta / (1.0 - specular_weight)
    }

    fn evaluate_specular_fast(
        &self,
        view: Vec3,
        normal: Vec3,
        sample: Vec3,
        cos_theta: f32,
        d_term: f32,
        specular_weight: f32,
        ks: Vec3,
    ) -> Spectrum {
        let g_term = util::geometry_smith_schlick_ggx(normal, view, sample, self.roughness);
        let specular_numerator = d_term * g_term * ks;
        let specular_denominator = 4.0 * normal.dot(view).max(0.0) * cos_theta;
        let specular = specular_numerator / specular_denominator.max(util::EPS);
        specular * cos_theta / specular_weight
    }

    fn pdf_diffuse_fast(&self, cos_theta: f32) -> f32 {
        cos_theta / f32::PI()
    }

    fn pdf_specular_fast(
        &self,
        view_direction: Vec3,
        normal: Vec3,
        halfway: Vec3,
        d_term: f32,
    ) -> f32 {
        (d_term * normal.dot(halfway)) / (4.0 * view_direction.dot(halfway))
    }
}

impl BSDF for PBR {
    fn evaluate(&self, view: Vec3, normal: Vec3, sample: Vec3, lobe_type: Lobe) -> Spectrum {
        let approx_fresnel =
            util::fresnel_schlick_scalar(1.0, DIELECTRIC_IOR, normal.dot(view).max(0.0));
        let mut specular_weight = util::lerp(approx_fresnel, 1.0, self.metallic);
        if specular_weight != 0.0 && specular_weight != 1.0 {
            specular_weight = specular_weight.clamp(self.clamp_weight.x, self.clamp_weight.y);
        }

        let cos_theta = normal.dot(sample).max(0.0);
        let halfway = (view + sample).normalize();

        let f0 = Vec3::splat(DIELECTRIC_F0).lerp(self.albedo, self.metallic);
        let ks = util::fresnel_schlick(halfway.dot(view).max(0.0), f0);

        if lobe_type == Lobe::DiffuseReflection {
            self.evaluate_diffuse_fast(cos_theta, specular_weight, ks)
        } else {
            let d_term = util::ggx_distribution(normal, halfway, self.roughness);
            self.evaluate_specular_fast(
                view,
                normal,
                sample,
                cos_theta,
                d_term,
                specular_weight,
                ks,
            )
        }
    }

    fn sample(&self, view: Vec3, normal: Vec3, rng: &mut RngState) -> BSDFSample {
        let rng_sample = rng.gen_r3();

        let approx_fresnel =
            util::fresnel_schlick_scalar(1.0, DIELECTRIC_IOR, normal.dot(view).max(0.0));
        let mut specular_weight = util::lerp(approx_fresnel, 1.0, self.metallic);
        // Clamp specular weight to prevent firelies. See Jakub Boksansky and Adam Marrs in RT gems 2 chapter 14.
        if specular_weight != 0.0 && specular_weight != 1.0 {
            specular_weight = specular_weight.clamp(self.clamp_weight.x, self.clamp_weight.y);
        }

        let (direction, lobe) = if rng_sample.z >= specular_weight {
            let (up, nt, nb) = util::create_cartesian(normal);
            let sample = util::cos_hemisphere(rng_sample.x, rng_sample.y);
            let sampled_direction = Vec3::new(
                sample.x * nb.x + sample.y * up.x + sample.z * nt.x,
                sample.x * nb.y + sample.y * up.y + sample.z * nt.y,
                sample.x * nb.z + sample.y * up.z + sample.z * nt.z,
            )
            .normalize();
            (sampled_direction, Lobe::DiffuseReflection)
        } else {
            let reflection_direction = util::reflect(-view, normal);
            let sampled_direction =
                util::sample_ggx(rng_sample.x, rng_sample.y, reflection_direction, self.roughness);
            (sampled_direction, Lobe::SpecularReflection)
        };

        let cos_theta = normal.dot(direction).max(util::EPS);
        let halfway = (view + direction).normalize();

        let f0 = Vec3::splat(DIELECTRIC_F0).lerp(self.albedo, self.metallic);
        let ks = util::fresnel_schlick(halfway.dot(view).max(0.0), f0);

        let (direction, lobe, pdf, spectrum) = if lobe == Lobe::DiffuseReflection {
            let pdf = self.pdf_diffuse_fast(cos_theta);
            let spectrum = self.evaluate_diffuse_fast(cos_theta, specular_weight, ks);
            (direction, Lobe::DiffuseReflection, pdf, spectrum)
        } else {
            let d_term = util::ggx_distribution(normal, halfway, self.roughness);
            let pdf = self.pdf_specular_fast(view, normal, halfway, d_term);
            let spectrum = self.evaluate_specular_fast(
                view,
                normal,
                direction,
                cos_theta,
                d_term,
                specular_weight,
                ks,
            );
            (direction, Lobe::SpecularReflection, pdf, spectrum)
        };

        BSDFSample { pdf, lobe, spectrum, direction }
    }

    fn pdf(&self, view: Vec3, normal: Vec3, sample: Vec3, lobe_type: Lobe) -> f32 {
        if lobe_type == Lobe::DiffuseReflection {
            let cos_theta = normal.dot(sample).max(0.0);
            self.pdf_diffuse_fast(cos_theta)
        } else {
            let halfway = (view + sample).normalize();
            let d_term = util::ggx_distribution(normal, halfway, self.roughness);
            self.pdf_specular_fast(view, normal, halfway, d_term)
        }
    }
}

pub fn get_pbr_bsdf(
    config: &TracingConfig,
    material: &MaterialData,
    uv: Vec2,
    atlas: &Image!(2D, type=f32, sampled),
    sampler: &Sampler,
) -> PBR {
    let albedo = if material.has_albedo_texture() {
        let scaled_uv = material.albedo.xy() + uv * material.albedo.zw();
        let albedo = atlas.sample_by_lod(*sampler, scaled_uv, 0.0);
        albedo.xyz()
    } else {
        material.albedo.xyz()
    };
    let roughness = if material.has_roughness_texture() {
        let scaled_uv = material.roughness.xy() + uv * material.roughness.zw();
        let roughness = atlas.sample_by_lod(*sampler, scaled_uv, 0.0);
        roughness.x
    } else {
        material.roughness.x
    };
    let metallic = if material.has_metallic_texture() {
        let scaled_uv = material.metallic.xy() + uv * material.metallic.zw();
        let metallic = atlas.sample_by_lod(*sampler, scaled_uv, 0.0);
        metallic.x
    } else {
        material.metallic.x
    };

    // Clamp values to avoid NaNs :P
    let roughness = roughness.max(util::EPS);
    let metallic = metallic.min(1.0 - util::EPS);

    PBR { albedo, roughness, metallic, clamp_weight: Vec2::new(0.1, 0.9) }
}
