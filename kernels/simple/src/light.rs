#[allow(dead_code)]
use spirv_std::num_traits::Float;
use {
    crate::{
        bsdf::{BSDFSample, Lobe, BSDF},
        inter::{BVHReference, Trace},
        rng::RngState,
        util,
    },
    shared::{LightPick, MaterialData, PerVertexData},
    spirv_std::glam::{UVec4, Vec3, Vec4Swizzles},
};

pub fn pdf(area: f32, len: f32, norm: Vec3, dir: Vec3) -> f32 {
    let cos_theta = norm.dot(-dir);
    if cos_theta <= 0.0 {
        return 0.0;
    }
    len.powi(2) / (area * cos_theta)
}

fn reflect(i: Vec3, n: Vec3) -> Vec3 {
    i - 2.0 * n.dot(i) * n
}

pub fn pick_light(table: &[LightPick], rng_state: &mut RngState) -> (u32, f32, f32) {
    let rng = rng_state.gen_r2();
    let entry = table[(rng.x * table.len() as f32) as usize];
    if rng.y < entry.ratio {
        (entry.triangle_index_a, entry.triangle_area_a, entry.triangle_pick_pdf_a)
    } else {
        (entry.triangle_index_b, entry.triangle_area_b, entry.triangle_pick_pdf_b)
    }
}

// https://www.cs.princeton.edu/~funk/tog02.pdf equation 1
pub fn pick_triangle_point(a: Vec3, b: Vec3, c: Vec3, rng_state: &mut RngState) -> Vec3 {
    let rng = rng_state.gen_r2();
    let r1_sqrt = rng.x.sqrt();
    (1.0 - r1_sqrt) * a + (r1_sqrt * (1.0 - rng.y)) * b + (r1_sqrt * rng.y) * c
}

pub fn calculate_light_pdf(
    light_area: f32,
    light_distance: f32,
    light_normal: Vec3,
    light_direction: Vec3,
) -> f32 {
    let cos_theta = light_normal.dot(-light_direction);
    if cos_theta <= 0.0 {
        return 0.0;
    }
    light_distance.powi(2) / (light_area * cos_theta)
}

#[derive(Default, Copy, Clone)]
pub struct LightSample {
    pub area: f32,
    pub normal: Vec3,
    pub pick_pdf: f32,
    pub emission: Vec3,
    pub triangle_idx: u32,
    pub throughput: Vec3,
    pub contribution: Vec3,
}

pub fn sample_direct_lighting(
    indices: &[UVec4],
    per_vertex: &[PerVertexData],
    materials: &[MaterialData],
    lights: &[LightPick],
    bvh: &BVHReference,
    throughput: Vec3,
    surface_bsdf: &impl BSDF,
    surface_point: Vec3,
    surface_normal: Vec3,
    ray_direction: Vec3,
    rng_state: &mut RngState,
) -> LightSample {
    // If the first entry is a sentinel, there are no lights
    if lights[0].is_sentinel() {
        return LightSample::default();
    }

    // Pick a light, get its surface properties
    let (light_index, area, pick_pdf) = pick_light(&lights, rng_state);
    let triangle = indices[light_index as usize];
    let vert_a = per_vertex[triangle.x as usize].vertex.xyz();
    let vert_b = per_vertex[triangle.y as usize].vertex.xyz();
    let vert_c = per_vertex[triangle.z as usize].vertex.xyz();
    let norm_a = per_vertex[triangle.x as usize].normal.xyz();
    let norm_b = per_vertex[triangle.y as usize].normal.xyz();
    let norm_c = per_vertex[triangle.z as usize].normal.xyz();
    let normal = (norm_a + norm_b + norm_c) / 3.0; // lights can use flat shading, no need to pay for interpolation
    let light_material = materials[triangle.w as usize];
    let emission = light_material.emissive.xyz();

    // Pick a point on the light
    let light_point = pick_triangle_point(vert_a, vert_b, vert_c, rng_state);
    let light_direction_unorm = light_point - surface_point;
    let light_distance = light_direction_unorm.length();
    let light_direction = light_direction_unorm / light_distance;

    // Sample the light directly using MIS
    let mut direct = Vec3::ZERO;
    let light_trace = bvh.intersect_any(
        per_vertex,
        indices,
        surface_point + light_direction * util::EPS,
        light_direction,
        light_distance - util::EPS * 2.0,
    );
    if !light_trace.hit {
        // Calculate light pdf for this sample
        let light_pdf = calculate_light_pdf(area, light_distance, normal, light_direction);
        if light_pdf > 0.0 {
            // Calculate BSDF attenuation for this sample
            let bsdf_attenuation = surface_bsdf.evaluate(
                -ray_direction,
                surface_normal,
                light_direction,
                Lobe::DiffuseReflection,
            );
            // Calculate BSDF pdf for this sample
            let bsdf_pdf = surface_bsdf.pdf(
                -ray_direction,
                surface_normal,
                light_direction,
                Lobe::DiffuseReflection,
            );
            if bsdf_pdf > 0.0 {
                // MIS - add the weighted sample
                let weight = get_weight(light_pdf, bsdf_pdf);
                direct = (bsdf_attenuation * emission * weight / light_pdf) / pick_pdf;
            }
        }
    }

    LightSample {
        area,
        normal,
        pick_pdf,
        emission,
        triangle_idx: light_index,
        throughput,
        contribution: throughput * direct,
    }
}

pub fn get_weight(p1: f32, p2: f32) -> f32 {
    util::power_heuristic(p1, p2)
}

pub fn calculate_bsdf_mis_contribution(
    trace: &Trace,
    bsdf_sample: &BSDFSample,
    light_sample: &LightSample,
) -> Vec3 {
    // If we haven't hit the same light as we sampled directly, no contribution
    if trace.triangle_index != light_sample.triangle_idx {
        return Vec3::ZERO;
    }

    // Calculate the light pdf for this sample
    let light_pdf = calculate_light_pdf(
        light_sample.area,
        trace.len,
        light_sample.normal,
        bsdf_sample.direction,
    );
    if light_pdf > 0.0 {
        // MIS - add the weighted sample
        let weight = get_weight(bsdf_sample.pdf, light_pdf);
        let direct = (bsdf_sample.spectrum * light_sample.emission * weight / bsdf_sample.pdf)
            / light_sample.pick_pdf;
        light_sample.throughput * direct
    } else {
        Vec3::ZERO
    }
}
