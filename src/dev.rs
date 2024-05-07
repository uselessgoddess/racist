use {
    crate::{Dtype, Vec3},
    num_traits::{cast, Float},
    rand::Rng,
    std::fmt::Debug,
};

#[derive(Debug, Clone, Copy)]
pub struct Glass<F> {
    pub ior: F,
}

#[derive(Debug, Clone, Copy)]
pub struct Light<F> {
    pub hv: F,
}

#[derive(Debug, Clone, Copy)]
pub enum MaterialKind<F> {
    Diffuse,
    Mirror,
    Glass(Glass<F>),
    Light(Light<F>),
}

#[derive(Debug, Clone, Copy)]
pub struct Material<F> {
    pub kind: MaterialKind<F>,
    pub rgb: Vec3<F>,
}

impl<F> Material<F> {
    pub fn diffuse(rgb: impl Into<Vec3<F>>) -> Self {
        Self { kind: MaterialKind::Diffuse, rgb: rgb.into() }
    }

    pub fn mirror(rgb: impl Into<Vec3<F>>) -> Self {
        Self { kind: MaterialKind::Mirror, rgb: rgb.into() }
    }

    pub fn glass(rgb: impl Into<Vec3<F>>, ior: F) -> Self {
        Self { kind: MaterialKind::Glass(Glass { ior }), rgb: rgb.into() }
    }

    pub fn light(rgb: impl Into<Vec3<F>>, hv: F) -> Self {
        Self { kind: MaterialKind::Light(Light { hv }), rgb: rgb.into() }
    }

    pub fn is_emission(&self) -> bool {
        matches!(self.kind, MaterialKind::Light(..))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Ray<F> {
    pub ori: Vec3<F>,
    pub dir: Vec3<F>,
}

#[derive(Debug, Clone, Copy)]
pub struct Hit<F> {
    pub pos: Vec3<F>,
    pub len: F,
    pub normal: Vec3<F>,
    pub obj_idx: usize,
}

pub trait Hitee<F> {
    fn shoot_at(&self, ray: Ray<F>, t_min: F, t_max: F) -> Option<Hit<F>>;
}

pub enum Interaction<F> {
    Scatter { dir: Vec3<F>, att: Vec3<F> },
    Emit { emission: Vec3<F> },
}

pub trait Surface<F> {
    fn surface_point<R: Rng>(&self, rng: &mut R) -> Vec3<F>;
    fn normal_at(&self, point: Vec3<F>) -> Vec3<F>;
}

pub struct Fov<F>(F);

impl<F: Dtype> Fov<F> {
    pub fn rad(rad: F) -> Self {
        Self(rad)
    }

    pub fn deg(deg: F) -> Self {
        let halfpi = F::zero().acos();
        let ninety = F::from_u8(90u8).unwrap();
        Self(deg * halfpi / ninety)
    }

    pub fn radians(self) -> F {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Transform<F> {
    pub scale: F,
    pub offset: F,
}

impl<F: Dtype> Transform<F> {
    pub fn apply(&self, x: F) -> F {
        self.scale * x + self.offset
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Camera<F> {
    pub(crate) pos: Vec3<F>,
    pub(crate) x: Transform<F>,
    pub(crate) y: Transform<F>,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl<F: Dtype> Camera<F> {
    pub fn new(fov: Fov<F>, (width, height): (usize, usize)) -> Self {
        let two = F::from_f64(2.0f64).unwrap();
        let (w, h) = (F::from_usize(width).unwrap(), F::from_usize(height).unwrap());
        let half_fov = (fov.radians() / two).tan();
        Self {
            pos: Vec3::new(F::zero(), F::zero(), F::zero()),
            x: Transform { scale: two * (w / h) * half_fov / w, offset: -(w / h) * half_fov },
            y: Transform { scale: -two * half_fov / h, offset: half_fov },
            width,
            height,
        }
    }

    pub fn at(mut self, x: F, y: F, z: F) -> Self {
        self.pos = Vec3::new(x, y, z);
        self
    }

    pub(crate) fn blank(&self) -> Vec<Vec3<F>> {
        vec![Vec3::new(F::zero(), F::zero(), F::zero()); self.width * self.height]
    }

    pub(crate) fn ray_through(&self, x_screen: F, y_screen: F) -> Ray<F> {
        Ray {
            ori: self.pos,
            dir: Vec3::new(self.x.apply(x_screen), self.y.apply(y_screen), -F::one()).normalize(),
        }
    }
}
