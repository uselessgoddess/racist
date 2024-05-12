#![no_std]

use {
    bytemuck::{Pod, Zeroable},
    glam::{Vec3, Vec4, Vec4Swizzles},
    spirv_std::glam::Vec2,
};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TracingConfig {
    pub cam_pos: Vec4,
    pub cam_rot: Vec4,
    pub width: u32,
    pub height: u32,
    pub min_bounces: u32,
    pub max_bounces: u32,
}

impl TracingConfig {
    pub fn soft() -> Self {
        Self {
            width: 1280,
            height: 720,
            cam_pos: Vec4::new(0.0, 1.0, -5.0, 0.0),
            cam_rot: Vec4::ZERO,
            min_bounces: 3,
            max_bounces: 4,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
pub struct PerVertexData {
    pub vertex: Vec4,
    pub normal: Vec4,
    pub tangent: Vec4,
    pub uv0: Vec2,
    pub uv1: Vec2,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BVHNode {
    aabb_min: Vec4, // w = triangle count
    aabb_max: Vec4, // w = left_node if triangle_count is 0, first_triangle_index if triangle_count is 1
}

impl Default for BVHNode {
    fn default() -> Self {
        Self {
            aabb_min: Vec4::new(f32::MAX, f32::MAX, f32::MAX, 0.0),
            aabb_max: Vec4::new(f32::MIN, f32::MIN, f32::MIN, 0.0),
        }
    }
}

impl BVHNode {
    // Immutable access
    pub fn triangle_count(&self) -> u32 {
        unsafe { core::mem::transmute(self.aabb_min.w) }
    }

    pub fn left_node_index(&self) -> u32 {
        unsafe { core::mem::transmute(self.aabb_max.w) }
    }

    pub fn right_node_index(&self) -> u32 {
        self.left_node_index() + 1
    }

    pub fn first_triangle_index(&self) -> u32 {
        unsafe { core::mem::transmute(self.aabb_max.w) }
    }

    pub fn aabb_min(&self) -> Vec3 {
        self.aabb_min.xyz()
    }

    pub fn aabb_max(&self) -> Vec3 {
        self.aabb_max.xyz()
    }

    pub fn is_leaf(&self) -> bool {
        self.triangle_count() > 0
    }

    // Mutable access
    pub fn set_triangle_count(&mut self, triangle_count: u32) {
        self.aabb_min.w = unsafe { core::mem::transmute(triangle_count) };
    }

    pub fn set_left_node_index(&mut self, left_node_index: u32) {
        self.aabb_max.w = unsafe { core::mem::transmute(left_node_index) };
    }

    pub fn set_first_triangle_index(&mut self, first_triangle_index: u32) {
        self.aabb_max.w = unsafe { core::mem::transmute(first_triangle_index) };
    }

    pub fn set_aabb_min(&mut self, aabb_min: Vec3) {
        self.aabb_min.x = aabb_min.x;
        self.aabb_min.y = aabb_min.y;
        self.aabb_min.z = aabb_min.z;
    }

    pub fn set_aabb_max(&mut self, aabb_max: Vec3) {
        self.aabb_max.x = aabb_max.x;
        self.aabb_max.y = aabb_max.y;
        self.aabb_max.z = aabb_max.z;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
pub struct MaterialData {
    // each Vec4 is either a color or an atlas location
    pub emissive: Vec4,
    pub albedo: Vec4,
    pub roughness: Vec4,
    pub metallic: Vec4,
    pub normals: Vec4,
    has_albedo_texture: u32,
    has_metallic_texture: u32,
    has_roughness_texture: u32,
    has_normal_texture: u32,
}

impl MaterialData {
    pub fn has_albedo_texture(&self) -> bool {
        self.has_albedo_texture != 0
    }

    pub fn set_has_albedo_texture(&mut self, has_albedo_texture: bool) {
        self.has_albedo_texture = if has_albedo_texture { 1 } else { 0 };
    }

    pub fn has_metallic_texture(&self) -> bool {
        self.has_metallic_texture != 0
    }

    pub fn set_has_metallic_texture(&mut self, has_metallic_texture: bool) {
        self.has_metallic_texture = if has_metallic_texture { 1 } else { 0 };
    }

    pub fn has_roughness_texture(&self) -> bool {
        self.has_roughness_texture != 0
    }

    pub fn set_has_roughness_texture(&mut self, has_roughness_texture: bool) {
        self.has_roughness_texture = if has_roughness_texture { 1 } else { 0 };
    }

    pub fn has_normal_texture(&self) -> bool {
        self.has_normal_texture != 0
    }

    pub fn set_has_normal_texture(&mut self, has_normal_texture: bool) {
        self.has_normal_texture = if has_normal_texture { 1 } else { 0 };
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
pub struct LightPick {
    pub triangle_index_a: u32,
    pub triangle_area_a: f32,
    pub triangle_pick_pdf_a: f32,
    pub triangle_index_b: u32,
    pub triangle_area_b: f32,
    pub triangle_pick_pdf_b: f32,
    pub ratio: f32,
}

// wgpu doesn't allow 0-sized buffers, so we use negative ratios to indicate sentinel values
impl LightPick {
    pub fn is_sentinel(&self) -> bool {
        self.ratio < 0.0
    }
}

#[cfg(target_arch = "spirv")]
pub mod polyfill {
    pub use spirv_std::{Image, Sampler};
}

#[cfg(not(target_arch = "spirv"))]
pub mod polyfill {
    use {
        core::marker::PhantomData,
        glam::{IVec2, Vec2, Vec4},
    };

    #[derive(Clone, Copy)]
    pub struct Sampler;

    pub struct Image<'a, A, B, C, D, E, F> {
        _phantom: PhantomData<(A, B, C, D, E, F)>,
        width: u32,
        height: u32,
        buffer: &'a [Vec4],
    }

    impl<'a, A> Image<'a, A, A, A, A, A, A> {
        pub const fn new(buffer: &'a [Vec4], width: u32, height: u32) -> Self {
            Image { _phantom: PhantomData, width, height, buffer }
        }

        fn sample_raw(&self, coord: IVec2) -> Vec4 {
            let x = coord.x as usize % self.width as usize;
            let y = coord.y as usize % self.height as usize;
            self.buffer[y * self.width as usize + x]
        }

        pub fn sample_by_lod(&self, _sampler: Sampler, coord: Vec2, _lod: f32) -> Vec4 {
            let scaled_uv = coord * Vec2::new(self.width as f32, self.height as f32);
            let frac_uv = scaled_uv.fract();
            let ceil_uv = scaled_uv.ceil().as_ivec2();
            let floor_uv = scaled_uv.floor().as_ivec2();

            // Bilinear filtering
            let c00 = self.sample_raw(floor_uv);
            let c01 = self.sample_raw(IVec2::new(floor_uv.x, ceil_uv.y));
            let c10 = self.sample_raw(IVec2::new(ceil_uv.x, floor_uv.y));
            let c11 = self.sample_raw(ceil_uv);
            let tx = frac_uv.x;
            let ty = frac_uv.y;

            let a = c00.lerp(c10, tx);
            let b = c01.lerp(c11, tx);
            a.lerp(b, ty)
        }
    }

    #[macro_export]
    macro_rules! Image {
        ($a:expr, $b:ident=$d:ident, $c:expr) => { Image<(), (), (), (), (), ()> };
    }

    pub type CpuImage<'fw> = Image<'fw, (), (), (), (), (), ()>;
}

pub use polyfill::{Image, Sampler};
