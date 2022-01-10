use crate::linalg::Three;

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Three<f64>,
    pub direction: Three<f64>,
}

impl Default for Ray {
    fn default() -> Self {
        Self {
            origin: Default::default(),
            direction: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Hit {
    pub position: Three<f64>,
    pub distance: f64,
    pub normal: Three<f64>,
}

impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.distance.eq(&other.distance)
    }
}

impl Eq for Hit {}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for Hit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

pub trait CanHit {
    fn hit_by(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit>;
}

#[derive(Debug, Clone, Copy)]
pub enum Interaction {
    Bounced { attenuation: Three<f64> },
    Absorbed { emission: Three<f64> },
    Missed,
}
