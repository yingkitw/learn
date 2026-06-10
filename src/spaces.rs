use rand::RngCore;

pub trait Space {
    fn sample(&self, rng: &mut dyn RngCore) -> Vec<f32>;
    fn contains(&self, x: &[f32]) -> bool;
    fn shape(&self) -> Vec<usize>;
}

#[derive(Debug, Clone)]
pub struct Discrete {
    pub n: usize,
}

impl Discrete {
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl Space for Discrete {
    fn sample(&self, rng: &mut dyn RngCore) -> Vec<f32> {
        let v = (rng.next_u32() as usize) % self.n;
        vec![v as f32]
    }

    fn contains(&self, x: &[f32]) -> bool {
        x.len() == 1 && x[0] >= 0.0 && x[0] < self.n as f32 && x[0].fract() == 0.0
    }

    fn shape(&self) -> Vec<usize> {
        vec![self.n]
    }
}

#[derive(Debug, Clone)]
pub struct Box {
    pub low: Vec<f32>,
    pub high: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Box {
    pub fn new(low: Vec<f32>, high: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(low.len(), high.len());
        assert_eq!(
            low.len(),
            shape.iter().product::<usize>(),
            "shape product must match number of elements"
        );
        Self { low, high, shape }
    }
}

impl Space for Box {
    fn sample(&self, rng: &mut dyn RngCore) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.low.len());
        for i in 0..self.low.len() {
            let l = self.low[i];
            let h = self.high[i];
            let u = rng.next_u32() as f32 / u32::MAX as f32;
            result.push(l + u * (h - l));
        }
        result
    }

    fn contains(&self, x: &[f32]) -> bool {
        if x.len() != self.low.len() {
            return false;
        }
        x.iter()
            .zip(self.low.iter().zip(self.high.iter()))
            .all(|(v, (l, h))| v >= l && v <= h)
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn discrete_sample_and_contains() {
        let d = Discrete::new(5);
        let mut rng = StdRng::seed_from_u64(42);
        let s = d.sample(&mut rng);
        assert!(d.contains(&s));
        assert!(!d.contains(&[5.0]));
        assert!(!d.contains(&[-1.0]));
        assert!(!d.contains(&[2.5]));
    }

    #[test]
    fn box_sample_and_contains() {
        let b = Box::new(vec![0.0, -1.0], vec![1.0, 1.0], vec![2]);
        let mut rng = StdRng::seed_from_u64(42);
        let s = b.sample(&mut rng);
        assert!(b.contains(&s));
        assert!(!b.contains(&[2.0, 0.0]));
    }

    #[test]
    fn discrete_shape_returns_n() {
        let d = Discrete::new(10);
        assert_eq!(d.shape(), vec![10]);
    }

    #[test]
    fn box_wrong_length_rejected() {
        let b = Box::new(vec![0.0, 0.0], vec![1.0, 1.0], vec![2]);
        assert!(!b.contains(&[0.5]));
        assert!(!b.contains(&[0.5, 0.5, 0.5]));
    }

    #[test]
    fn discrete_zero_not_contains_negative() {
        let d = Discrete::new(3);
        assert!(!d.contains(&[-1.0]));
        assert!(!d.contains(&[3.0]));
        assert!(d.contains(&[0.0]));
        assert!(d.contains(&[2.0]));
    }
}
