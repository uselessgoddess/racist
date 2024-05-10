#![cfg_attr(target_arch = "spirv", no_std)]
// #![deny(warnings)]

mod rng;

use {
    crate::rng::RngState,
    core::ops::{Add, Div, Mul, Sub},
    shared::TracingConfig,
    spirv_std::{
        glam::{
            vec2, vec3, vec4, Mat2, UVec2, UVec3, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles,
        },
        num_traits::{Float, Pow},
        spirv,
    },
};

fn trace_pixel(id: UVec3, config: &TracingConfig, rng: UVec2) -> (Vec4, UVec2) {
    let mut rng_state = RngState::new(rng);

    let mut uv =
        Vec2::new(id.x as f32 / config.width as f32, 1.0 - id.y as f32 / config.height as f32)
            * 2.0
            - 1.0;
    uv.y *= config.height as f32 / config.width as f32;

    (vec3(1.0, 0.0, 0.5).extend(1.0), rng_state.next_state())
}

#[spirv(compute(threads(8, 8, 1)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] config: &TracingConfig,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rng: &mut [UVec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [Vec4],
) {
    let index = (id.y * config.width + id.x) as usize;
    let (pixel, state) = trace_pixel(id, config, rng[index]);

    output[index] += pixel;
    rng[index] = state;
}
