use crate::core::Transition;
use rand::RngCore;

pub struct ReplayBuffer<O, A> {
    buffer: Vec<Transition<O, A>>,
    capacity: usize,
    position: usize,
}

impl<O: Clone, A: Clone> ReplayBuffer<O, A> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            position: 0,
        }
    }

    pub fn add(&mut self, transition: Transition<O, A>) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(transition);
        } else {
            self.buffer[self.position] = transition;
        }
        self.position = (self.position + 1) % self.capacity;
    }

    pub fn sample(&self, batch_size: usize, rng: &mut dyn RngCore) -> Option<Vec<Transition<O, A>>> {
        if self.buffer.is_empty() {
            return None;
        }
        let actual = batch_size.min(self.buffer.len());
        let mut sampled = Vec::with_capacity(actual);
        for _ in 0..actual {
            let idx = (rng.next_u32() as usize) % self.buffer.len();
            sampled.push(self.buffer[idx].clone());
        }
        Some(sampled)
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_buffer_add_and_sample() {
        let mut buf: ReplayBuffer<f32, usize> = ReplayBuffer::new(10);
        let mut rng = StdRng::seed_from_u64(0);
        buf.add(Transition {
            obs: 0.0,
            action: 0,
            reward: 1.0,
            next_obs: 1.0,
            terminated: false,
            truncated: false,
        });
        let batch = buf.sample(1, &mut rng).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_sample_returns_none_when_empty() {
        let buf: ReplayBuffer<f32, usize> = ReplayBuffer::new(10);
        let mut rng = StdRng::seed_from_u64(0);
        assert!(buf.sample(1, &mut rng).is_none());
    }

    #[test]
    fn test_buffer_respects_capacity() {
        let mut buf: ReplayBuffer<f32, usize> = ReplayBuffer::new(2);
        for i in 0..5 {
            buf.add(Transition {
                obs: i as f32,
                action: i,
                reward: i as f32,
                next_obs: (i + 1) as f32,
                terminated: false,
                truncated: false,
            });
        }
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_buffer_batch_size_limited_by_length() {
        let mut buf: ReplayBuffer<f32, usize> = ReplayBuffer::new(10);
        let mut rng = StdRng::seed_from_u64(0);
        for i in 0..3 {
            buf.add(Transition {
                obs: i as f32,
                action: i,
                reward: i as f32,
                next_obs: (i + 1) as f32,
                terminated: false,
                truncated: false,
            });
        }
        let batch = buf.sample(100, &mut rng).unwrap();
        assert_eq!(batch.len(), 3);
    }
}
