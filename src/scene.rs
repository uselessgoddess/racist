use {
    crate::{shapes::Sphere, Dtype, Hit, Hitee, Material, Ray, Vec3},
    num_traits::Float,
    rand::Rng,
};

pub trait Tracer<F> {
    fn trace<R: Rng>(&self, ray: Ray<F>, scene: &Scene<F>, rng: &mut R) -> Option<Vec3<F>>;
}

type Object<F> = Sphere<F>;

pub struct Scene<F> {
    objects: Vec<Object<F>>,
    emissions: Vec<(usize, Object<F>)>,
    materials: Vec<Material<F>>,
    materials_sparse: Vec<MaterialIdx>,
}

impl<F> Scene<F>
where
    F: Clone,
{
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            emissions: Vec::new(),
            materials: Vec::new(),
            materials_sparse: Vec::new(),
        }
    }

    pub fn material<M: Into<Material<F>>>(&mut self, material: M) -> MaterialIdx {
        let idx = MaterialIdx(self.materials.len());
        self.materials.push(material.into());
        idx
    }

    pub fn object<O: Into<Object<F>>>(&mut self, object: O, mat_idx: MaterialIdx) -> &mut Self {
        let object = object.into();
        let idx = self.objects.len();

        self.objects.push(object.clone());
        self.materials_sparse.push(mat_idx);
        if self.material_for(idx).is_emission() {
            self.emissions.push((idx, object));
        }
        self
    }

    pub fn material_for(&self, obj_idx: usize) -> &Material<F> {
        &self.materials[self.materials_sparse[obj_idx].0]
    }

    pub fn emissions(&self) -> &[(usize, Object<F>)] {
        &self.emissions
    }
}

impl<F: Dtype> Hitee<F> for Scene<F> {
    fn shoot_at(&self, ray: Ray<F>, t_min: F, mut t_max: F) -> Option<Hit<F>> {
        let mut opt_hit = None;
        for (idx, obj) in self.objects.iter().enumerate() {
            if let Some(mut hit) = obj.shoot_at(ray, t_min, t_max) {
                if hit.len < t_max {
                    hit.obj_idx = idx;
                    opt_hit = Some(hit);
                    t_max = hit.len;
                }
            }
        }
        opt_hit
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterialIdx(usize);
