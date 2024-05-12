#![feature(let_chains)]
#![cfg_attr(target_arch = "spirv", no_std)]
// #![deny(warnings)]

mod bsdf;
mod inter;
mod light;
mod rng;
mod skybox;
mod util;
mod vec;

use {
    crate::{
        bsdf::{Lobe, BSDF},
        inter::{BVHReference, Trace},
        rng::RngState,
    },
    core::{
        cmp::Ordering,
        ops::{Add, Div, Mul, Sub},
    },
    shared::{BVHNode, LightPick, MaterialData, PerVertexData, Sampler, TracingConfig},
    spirv_std::{
        glam::{
            vec2, vec3, vec4, Mat2, Mat3, UVec2, UVec3, UVec4, Vec2, Vec3, Vec3Swizzles, Vec4,
            Vec4Swizzles,
        },
        num_traits::{Float, Pow},
        spirv, Image,
    },
};

pub const EPS: f32 = 0.001;

fn trace_pixel(
    id: UVec3,
    config: &TracingConfig,
    rng: UVec2,
    indices: &[UVec4],
    per_vertex: &[PerVertexData],
    nodes_buffer: &[BVHNode],
    materials: &[MaterialData],
    lights: &[LightPick],
    sampler: &Sampler,
    atlas: &Image!(2D, type=f32, sampled),
) -> (Vec4, UVec2) {
    let mut rng_state = RngState::new(rng);

    let suv = id.xy().as_vec2() + rng_state.gen_r2();
    let mut uv =
        Vec2::new(suv.x / config.width as f32, 1.0 - suv.y / config.height as f32) * 2.0 - 1.0;
    uv.y *= config.height as f32 / config.width as f32;

    let mut ori = config.cam_pos.xyz();
    let euler_mat =
        Mat3::from_rotation_y(config.cam_rot.y) * Mat3::from_rotation_x(config.cam_rot.x);
    let mut dir = euler_mat * (Vec3::new(uv.x, uv.y, 1.0).normalize());

    let mut throughput = Vec3::ONE;
    let mut radiance = Vec3::ZERO;
    let mut bsdf_sample = bsdf::BSDFSample::default();
    let mut light_sample = light::LightSample::default();

    let bvh = BVHReference { nodes: nodes_buffer };

    for bounce in 0..16 {
        let trace = bvh.intersect_nearest(per_vertex, indices, ori, dir);
        let hit = ori + dir * trace.len;

        if !trace.hit {
            let sun = Vec3::new(0.5, 1.3, 1.0).normalize().extend(15.0);
            radiance += throughput * skybox::scatter(sun, ori, dir);
            break;
        } else {
            let material = materials[trace.triangle.w as usize];

            if material.emissive.xyz() != Vec3::ZERO {
                if trace.backface {
                    break; // Break since emissives don't bounce light
                }

                if bounce == 0 || bsdf_sample.lobe != Lobe::DiffuseReflection {
                    radiance += util::mask_nan(throughput * material.emissive.xyz() * 15.0);
                    break;
                }

                if bsdf_sample.lobe == Lobe::DiffuseReflection {
                    let direct_contribution =
                        light::calculate_bsdf_mis_contribution(&trace, &bsdf_sample, &light_sample);
                    radiance += util::mask_nan(direct_contribution);
                    break;
                }
            }

            let vertex_data_a = per_vertex[trace.triangle.x as usize];
            let vertex_data_b = per_vertex[trace.triangle.y as usize];
            let vertex_data_c = per_vertex[trace.triangle.z as usize];
            let vert_a = vertex_data_a.vertex.xyz();
            let vert_b = vertex_data_b.vertex.xyz();
            let vert_c = vertex_data_c.vertex.xyz();
            let norm_a = vertex_data_a.normal.xyz();
            let norm_b = vertex_data_b.normal.xyz();
            let norm_c = vertex_data_c.normal.xyz();
            let uv_a = vertex_data_a.uv0;
            let uv_b = vertex_data_b.uv0;
            let uv_c = vertex_data_c.uv0;
            let bary = util::barycentric(hit, vert_a, vert_b, vert_c);
            let mut norm = bary.x * norm_a + bary.y * norm_b + bary.z * norm_c;
            let mut uv = bary.x * uv_a + bary.y * uv_b + bary.z * uv_c;
            if uv.clamp(Vec2::ZERO, Vec2::ONE) != uv {
                uv = uv.fract(); // wrap UVs
            }

            if material.has_normal_texture() {
                let scaled_uv = material.normals.xy() + uv * material.normals.zw();
                let normal_map = atlas.sample_by_lod(*sampler, scaled_uv, 0.0) * 2.0 - 1.0;
                let tangent_a = vertex_data_a.tangent.xyz();
                let tangent_b = vertex_data_b.tangent.xyz();
                let tangent_c = vertex_data_c.tangent.xyz();
                let tangent = bary.x * tangent_a + bary.y * tangent_b + bary.z * tangent_c;
                let tbn = Mat3::from_cols(tangent, tangent.cross(norm), norm);
                norm = (tbn * normal_map.xyz()).normalize();
            }

            let bsdf = bsdf::get_pbr_bsdf(config, &material, uv, atlas, sampler);
            // let bsdf = bsdf::Lambertian { albedo: col };
            // let bsdf = bsdf::Glass { albedo: col, ior: 1.5, roughness: 0.7 };

            bsdf_sample = bsdf.sample(-dir, norm, &mut rng_state);

            if bsdf_sample.lobe == Lobe::DiffuseReflection {
                light_sample = light::sample_direct_lighting(
                    indices,
                    per_vertex,
                    materials,
                    lights,
                    &bvh,
                    throughput,
                    &bsdf,
                    hit,
                    norm,
                    dir,
                    &mut rng_state,
                );
                radiance += util::mask_nan(light_sample.contribution);
            }

            throughput *= bsdf_sample.spectrum / bsdf_sample.pdf;
            dir = bsdf_sample.direction;
            ori = hit + dir * EPS;

            if bounce > 8 {
                let prob = throughput.max_element();
                if rng_state.gen_r1() > prob {
                    break;
                }
                throughput *= 1.0 / prob;
            }
        }
    }

    (radiance.extend(1.0), rng_state.next_state())
}

#[spirv(compute(threads(8, 8, 1)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] config: &TracingConfig,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rng: &mut [UVec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] index_buffer: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] per_vertex_buffer: &[PerVertexData],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] nodes_buffer: &[BVHNode],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] materials: &[MaterialData],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] lights: &[LightPick],
    #[spirv(descriptor_set = 0, binding = 8)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 9)] atlas: &Image!(2D, type=f32, sampled),
) {
    let index = (id.y * config.width + id.x) as usize;
    let (pixel, state) = trace_pixel(
        id,
        config,
        rng[index],
        index_buffer,
        per_vertex_buffer,
        nodes_buffer,
        materials,
        lights,
        sampler,
        atlas,
    );

    output[index] += pixel;
    rng[index] = state;
}
