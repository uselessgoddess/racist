pub mod att;
pub mod dev;
mod rendering;
pub mod scene;
pub mod shapes;
pub mod tracer;

pub use {
    dev::{
        Camera, Fov, Glass, Hit, Hitee, Interaction, Light, Material, MaterialKind, Ray, Surface,
    },
    rendering::render,
    scene::{Scene, Tracer},
    shapes::Sphere,
};
// pub use shapes::{Plane, Prism, Sphere, Triangle};
pub use tracer::PathTracer;

pub(crate) use {na::Vector3 as Vec3, nalgebra as na, num_traits as nt};

pub trait Unit:
    'static
    + Copy
    + Clone
    + Default
    + std::fmt::Debug
    + PartialEq
    + PartialOrd
    + Send
    + Sync
    + Unpin
    + nalgebra::Scalar
{
}

impl Unit for f32 {}
impl Unit for f64 {}

pub trait Dtype:
    Unit
    + nalgebra::RealField
    + nalgebra::ClosedAdd
    + nalgebra::ClosedSub
    + nalgebra::ClosedMul
    + nalgebra::ClosedDiv
    + num_traits::ToPrimitive
{
}

impl<All> Dtype for All where
    All: Unit
        + nalgebra::RealField
        + nalgebra::ClosedAdd
        + nalgebra::ClosedSub
        + nalgebra::ClosedMul
        + nalgebra::ClosedDiv
        + num_traits::ToPrimitive
{
}
