use crate::{material::Material, objects::Object};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterialIdx(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ObjectIdx(pub usize);

pub struct Scene {
    objects: Vec<Object>,
    object_material_idx: Vec<MaterialIdx>,
    materials: Vec<Material>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            object_material_idx: Vec::new(),
            materials: Vec::new(),
        }
    }

    pub fn add_material(&mut self, material: Material) -> MaterialIdx {
        let idx = MaterialIdx(self.materials.len());
        self.materials.push(material);
        idx
    }

    pub fn add_object<O: Into<Object>>(&mut self, obj: O, mat_idx: MaterialIdx) -> ObjectIdx {
        let obj_idx = ObjectIdx(self.objects.len());
        self.objects.push(obj.into());
        self.object_material_idx.push(mat_idx);
        obj_idx
    }

    pub fn material_for(&self, obj_idx: ObjectIdx) -> &Material {
        let mat_idx = self.object_material_idx[obj_idx.0];
        &self.materials[mat_idx.0]
    }

    pub fn objects(&self) -> &Vec<Object> {
        &self.objects
    }
}