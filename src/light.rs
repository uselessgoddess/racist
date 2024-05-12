use {
    glam::{UVec4, Vec3, Vec4, Vec4Swizzles},
    rand::Rng,
    shared::{LightPick, MaterialData},
};

fn triangle_area(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let side_a = b - a;
    let side_b = c - b;
    let side_c = a - c;
    let s = (side_a.length() + side_b.length() + side_c.length()) / 2.0;
    (s * (s - side_a.length()) * (s - side_b.length()) * (s - side_c.length())).sqrt()
}

pub fn compute_emissive_mask(indices: &[UVec4], material_datas: &[MaterialData]) -> Vec<bool> {
    let mut emissive_mask = vec![false; indices.len()];
    for i in 0..indices.len() {
        if material_datas[indices[i].w as usize].emissive.xyz() != Vec3::ZERO {
            emissive_mask[i] = true;
        }
    }
    emissive_mask
}

// NOTE: `mask` indicates which triangles are valid for picking
pub fn build_light_pick_table(
    vertices: &[Vec4],
    indices: &[UVec4],
    mask: &[bool],
    material_datas: &[MaterialData],
) -> Vec<LightPick> {
    // Calculate areas and probabilities of picking each triangle
    let mut triangle_areas = vec![0.0; indices.len()];
    let mut triangle_powers = vec![0.0; indices.len()];
    let mut total_power = 0.0;
    let mut total_tris = 0;
    for i in 0..indices.len() {
        if !mask[i] {
            continue;
        }
        total_tris += 1;

        let triangle = indices[i];
        let a = vertices[triangle.x as usize].xyz();
        let b = vertices[triangle.y as usize].xyz();
        let c = vertices[triangle.z as usize].xyz();

        let triangle_area = triangle_area(a, b, c);
        triangle_areas[i] = triangle_area;

        let triangle_power =
            material_datas[triangle.w as usize].emissive.xyz().dot(Vec3::ONE) * triangle_area;
        triangle_powers[i] = triangle_power;
        total_power += triangle_power;
    }
    if total_tris == 0 {
        // If there are 0 entries, put in a stupid sentinel value
        return vec![LightPick { ratio: -1.0, ..Default::default() }];
    }
    let mut triangle_probabilities = vec![0.0; indices.len()];
    for i in 0..indices.len() {
        triangle_probabilities[i] = triangle_powers[i] / total_power;
    }
    let average_probability = triangle_probabilities.iter().sum::<f32>() / total_tris as f32;
    // Build histogram bins. Each entry contains 2 discrete outcomes.
    #[derive(Debug)]
    struct TriangleBin {
        index_a: usize,
        probability_a: f32,
        index_b: usize,
        probability_b: f32,
    }
    let mut bins = triangle_probabilities
        .iter()
        .enumerate()
        .map(|x| TriangleBin { index_a: x.0, probability_a: *x.1, index_b: 0, probability_b: 0.0 })
        .filter(|x| x.probability_a != 0.0)
        .collect::<Vec<_>>();
    bins.sort_by(|a, b| {
        a.probability_a.partial_cmp(&b.probability_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Robin hood - take from the most probable and give to the least probable
    let num_bins = bins.len();
    let mut most_probable = num_bins - 1;
    for i in 0..num_bins {
        let needed = average_probability - bins[i].probability_a;
        if needed <= 0.0 {
            break;
        }

        bins[i].index_b = bins[most_probable].index_a;
        bins[i].probability_b = needed;
        bins[most_probable].probability_a -= needed;
        if bins[most_probable].probability_a <= average_probability {
            most_probable -= 1;
        }
    }

    // Build the table
    let table = bins
        .iter()
        .map(|x| LightPick {
            triangle_index_a: x.index_a as u32,
            triangle_index_b: x.index_b as u32,
            triangle_pick_pdf_a: triangle_probabilities[x.index_a],
            triangle_area_a: triangle_areas[x.index_a],
            triangle_area_b: triangle_areas[x.index_b],
            triangle_pick_pdf_b: triangle_probabilities[x.index_b],
            ratio: x.probability_a / (x.probability_a + x.probability_b),
        })
        .collect::<Vec<_>>();

    table
}
