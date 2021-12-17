use crate::linalg::Vec3;
use crate::ray::{CanIntersect, Intersection, Material, Ray};

pub(crate) struct Sphere {
    pub(crate) center: Vec3,
    pub(crate) radius: f64,
    pub(crate) material: Material,
}

impl CanIntersect for Sphere {
    fn intersect(&self, ray: Ray) -> Option<Intersection> {
        let l = self.center - ray.origin;
        let adj = l.dot(&ray.direction);
        let d2 = l.dot(&l) - (adj * adj);
        let radius2 = self.radius * self.radius;
        if d2 > radius2 {
            return None;
        }
        let thc = (radius2 - d2).sqrt();
        let t0 = adj - thc;
        let t1 = adj + thc;

        if t0 < 0.0 && t1 < 0.0 {
            return None;
        }

        let distance = t0.min(t1);
        Some(Intersection {
            distance,
            result: Ray {
                origin: ray.origin + ray.direction * distance,
                direction: ray.direction, // TODO normal
            },
            material: self.material,
        })
    }
}
