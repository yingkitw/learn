use crate::core::Agent;
use crate::utils::epsilon_greedy;
use rand::rngs::StdRng;
use rand::RngCore;
use rand::SeedableRng;

pub struct QLearning {
    q_table: Vec<Vec<f32>>,
    alpha: f32,
    gamma: f32,
    epsilon: f32,
    rng: StdRng,
}

impl QLearning {
    pub fn new(n_states: usize, n_actions: usize, alpha: f32, gamma: f32, epsilon: f32, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut q_table = vec![vec![0.0; n_actions]; n_states];
        for s in 0..n_states {
            for a in 0..n_actions {
                q_table[s][a] = (rng.next_u32() as f32 / u32::MAX as f32) * 0.01;
            }
        }
        Self {
            q_table,
            alpha,
            gamma,
            epsilon,
            rng,
        }
    }

    pub fn q_values(&self, state: usize) -> &[f32] {
        &self.q_table[state]
    }
}

impl Agent for QLearning {
    type Observation = usize;
    type Action = usize;

    fn act(&mut self, obs: &Self::Observation, training: bool) -> Self::Action {
        let q = &self.q_table[*obs];
        if training {
            epsilon_greedy(q, self.epsilon, &mut self.rng)
        } else {
            q.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }
    }

    fn handle_step(
        &mut self,
        obs: &Self::Observation,
        action: &Self::Action,
        reward: f32,
        next_obs: &Self::Observation,
        done: bool,
    ) {
        let max_next = if done {
            0.0
        } else {
            self.q_table[*next_obs]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max)
        };
        let target = reward + self.gamma * max_next;
        let td_error = target - self.q_table[*obs][*action];
        self.q_table[*obs][*action] += self.alpha * td_error;
    }

    fn episode_end(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q_learning_update() {
        let mut agent = QLearning::new(2, 2, 0.1, 0.9, 0.0, 0);
        let initial = agent.q_values(0)[0];
        agent.handle_step(&0, &0, 1.0, &1, false);
        let updated = agent.q_values(0)[0];
        assert!(updated > initial);
    }
}
