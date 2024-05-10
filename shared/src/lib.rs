#![no_std]

use {
    bytemuck::{Pod, Zeroable},
    spirv_std::glam::Vec2,
};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TracingConfig {
    pub width: u32,
    pub height: u32,
}

impl TracingConfig {
    pub fn soft() -> Self {
        Self { width: 1280, height: 720 }
    }
}
