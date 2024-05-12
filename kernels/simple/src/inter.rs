#[allow(unused_imports)]
use spirv_std::num_traits::Float;
use {
    crate::vec::FixedVec,
    core::mem,
    shared::{BVHNode, PerVertexData},
    spirv_std::{
        glam::{UVec4, Vec3, Vec4, Vec4Swizzles},
        num_traits::Signed,
    },
};

// https://github.com/pema99/rust-path-tracer/blob/master/kernels/src/intersection.rs
// https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm
fn muller_trumbore(
    ro: Vec3,
    rd: Vec3,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    out_t: &mut f32,
    out_backface: &mut bool,
) -> bool {
    *out_t = 0.0;

    let edge1 = b - a;
    let edge2 = c - a;

    let pv = rd.cross(edge2);
    let det = edge1.dot(pv);
    *out_backface = det < 0.0;

    if det.abs() < 1e-6 {
        return false;
    }

    let inv_det = 1.0 / det;
    let tv = ro - a;

    let u = tv.dot(pv) * inv_det;
    if u < 0.0 || u > 1.0 {
        return false;
    }

    let qv = tv.cross(edge1);

    let v = rd.dot(qv) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }

    let t = edge2.dot(qv) * inv_det;
    if t < 0.0 {
        return false;
    }
    *out_t = t;

    return true;
}

pub struct Trace {
    pub triangle: UVec4,
    pub triangle_index: u32,
    pub len: f32,
    pub hit: bool,
    pub backface: bool,
}

impl Trace {
    pub fn miss() -> Self {
        Self { triangle: UVec4::ZERO, triangle_index: 0, len: 1e6, hit: false, backface: false }
    }
}

pub fn intersect_slow_as_shit(
    vertex_buffer: &[PerVertexData],
    index_buffer: &[UVec4],
    ro: Vec3,
    rd: Vec3,
) -> Trace {
    let mut result = Trace::miss();
    for i in 0..index_buffer.len() {
        let triangle = index_buffer[i];
        let a = vertex_buffer[triangle.x as usize].vertex.xyz();
        let b = vertex_buffer[triangle.y as usize].vertex.xyz();
        let c = vertex_buffer[triangle.z as usize].vertex.xyz();

        let mut t = 0.0;
        let mut backface = false;
        if muller_trumbore(ro, rd, a, b, c, &mut t, &mut backface) && t > 0.001 && t < result.len {
            result.triangle = triangle;
            result.triangle_index = i as u32;
            result.len = result.len.min(t);
            result.hit = true;
            result.backface = backface;
        }
    }
    result
}

fn intersect_aabb(aabb_min: Vec3, aabb_max: Vec3, ro: Vec3, rd: Vec3, prev_min_t: f32) -> f32 {
    let tx1 = (aabb_min.x - ro.x) / rd.x;
    let tx2 = (aabb_max.x - ro.x) / rd.x;
    let mut tmin = tx1.min(tx2);
    let mut tmax = tx1.max(tx2);
    let ty1 = (aabb_min.y - ro.y) / rd.y;
    let ty2 = (aabb_max.y - ro.y) / rd.y;
    tmin = tmin.max(ty1.min(ty2));
    tmax = tmax.min(ty1.max(ty2));
    let tz1 = (aabb_min.z - ro.z) / rd.z;
    let tz2 = (aabb_max.z - ro.z) / rd.z;
    tmin = tmin.max(tz1.min(tz2));
    tmax = tmax.min(tz1.max(tz2));
    if tmax >= tmin && tmax > 0.0 && tmin < prev_min_t {
        tmin
    } else {
        f32::MAX
    }
}

pub struct BVHReference<'a> {
    pub nodes: &'a [BVHNode],
}

impl<'a> BVHReference<'a> {
    pub fn intersect_nearest(
        &self,
        per_vertex_buffer: &[PerVertexData],
        index_buffer: &[UVec4],
        ro: Vec3,
        rd: Vec3,
    ) -> Trace {
        self.intersect_front_to_back::<true>(per_vertex_buffer, index_buffer, ro, rd, 0.0)
    }

    pub fn intersect_any(
        &self,
        per_vertex_buffer: &[PerVertexData],
        index_buffer: &[UVec4],
        ro: Vec3,
        rd: Vec3,
        max_t: f32,
    ) -> Trace {
        self.intersect_front_to_back::<false>(per_vertex_buffer, index_buffer, ro, rd, max_t)
    }

    fn intersect_front_to_back<const NEAREST: bool>(
        &self,
        per_vertex_buffer: &[PerVertexData],
        index_buffer: &[UVec4],
        ro: Vec3,
        rd: Vec3,
        max_t: f32,
    ) -> Trace {
        let mut stack = FixedVec::<usize, 32>::new();
        stack.push(0);

        let mut result = Trace::miss();
        while !stack.is_empty() {
            let node_index = stack.pop().unwrap();
            let node = &self.nodes[node_index];
            if node.is_leaf() {
                for i in 0..node.triangle_count() {
                    let triangle_index = node.first_triangle_index() + i;
                    let triangle = index_buffer[triangle_index as usize];
                    let a = per_vertex_buffer[triangle.x as usize].vertex.xyz();
                    let b = per_vertex_buffer[triangle.y as usize].vertex.xyz();
                    let c = per_vertex_buffer[triangle.z as usize].vertex.xyz();

                    let mut t = 0.0;
                    let mut backface = false;
                    if muller_trumbore(ro, rd, a, b, c, &mut t, &mut backface)
                        && t > 0.001
                        && t < result.len
                        && (NEAREST || t <= max_t)
                    {
                        result.triangle = triangle;
                        result.triangle_index = triangle_index;
                        result.len = result.len.min(t);
                        result.hit = true;
                        result.backface = backface;
                        if !NEAREST {
                            return result;
                        }
                    }
                }
            } else {
                // find closest child
                let mut min_index = node.left_node_index() as usize;
                let mut max_index = node.right_node_index() as usize;
                let mut min_child = &self.nodes[min_index];
                let mut max_child = &self.nodes[max_index];
                let mut min_dist =
                    intersect_aabb(min_child.aabb_min(), min_child.aabb_max(), ro, rd, result.len);
                let mut max_dist =
                    intersect_aabb(max_child.aabb_min(), max_child.aabb_max(), ro, rd, result.len);
                if min_dist > max_dist {
                    mem::swap(&mut min_index, &mut max_index);
                    mem::swap(&mut min_dist, &mut max_dist);
                    mem::swap(&mut min_child, &mut max_child);
                }

                // if min child isn't hit, both children aren't hit, so skip
                if min_dist.abs() == f32::MAX {
                    continue;
                }

                // push valid children in the best order
                if max_dist.abs() < f32::MAX {
                    stack.push(max_index);
                }
                stack.push(min_index); // <-- this child will be popped first
            }
        }

        result
    }
}
