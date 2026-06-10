use rand::RngCore;

pub fn epsilon_greedy(q_values: &[f32], epsilon: f32, rng: &mut dyn RngCore) -> usize {
    if (rng.next_u32() as f32 / u32::MAX as f32) < epsilon {
        (rng.next_u32() as usize) % q_values.len()
    } else {
        q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}

pub fn compute_returns(rewards: &[f32], gamma: f32) -> Vec<f32> {
    let mut returns = vec![0.0; rewards.len()];
    let mut g = 0.0;
    for i in (0..rewards.len()).rev() {
        g = rewards[i] + gamma * g;
        returns[i] = g;
    }
    returns
}

pub fn one_hot(index: usize, n: usize) -> Vec<f32> {
    let mut v = vec![0.0; n];
    if index < n {
        v[index] = 1.0;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_compute_returns() {
        let r = vec![1.0, 0.0, 1.0];
        let g = compute_returns(&r, 0.9);
        assert!((g[2] - 1.0).abs() < 1e-6);
        assert!((g[1] - 0.9).abs() < 1e-6);
        assert!((g[0] - 1.81).abs() < 1e-6);
    }

    #[test]
    fn test_one_hot() {
        assert_eq!(one_hot(2, 4), vec![0.0, 0.0, 1.0, 0.0]);
    }
}
