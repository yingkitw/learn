use crate::core::Agent;
use crate::utils::epsilon_greedy;
use rand::rngs::StdRng;
use rand::RngCore;
use rand::SeedableRng;

pub struct Sarsa {
    q_table: Vec<Vec<f32>>,
    alpha: f32,
    gamma: f32,
    epsilon: f32,
    rng: StdRng,
    prev_obs: Option<usize>,
    prev_action: Option<usize>,
    prev_reward: f32,
}

impl Sarsa {
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
            prev_obs: None,
            prev_action: None,
            prev_reward: 0.0,
        }
    }

    pub fn q_values(&self, state: usize) -> &[f32] {
        &self.q_table[state]
    }
}

impl Agent for Sarsa {
    type Observation = usize;
    type Action = usize;

    fn act(&mut self, obs: &Self::Observation, training: bool) -> Self::Action {
        let q = &self.q_table[*obs];
        let action = if training {
            epsilon_greedy(q, self.epsilon, &mut self.rng)
        } else {
            q.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        };

        if let (Some(prev_o), Some(prev_a)) = (self.prev_obs, self.prev_action) {
            let target = self.q_table[*obs][action];
            let td_error = self.prev_reward + self.gamma * target - self.q_table[prev_o][prev_a];
            self.q_table[prev_o][prev_a] += self.alpha * td_error;
        }

        self.prev_obs = Some(*obs);
        self.prev_action = Some(action);
        action
    }

    fn handle_step(
        &mut self,
        _obs: &Self::Observation,
        _action: &Self::Action,
        reward: f32,
        _next_obs: &Self::Observation,
        done: bool,
    ) {
        if done {
            if let (Some(prev_o), Some(prev_a)) = (self.prev_obs, self.prev_action) {
                let td_error = reward - self.q_table[prev_o][prev_a];
                self.q_table[prev_o][prev_a] += self.alpha * td_error;
            }
            self.prev_obs = None;
            self.prev_action = None;
        } else {
            self.prev_reward = reward;
        }
    }

    fn episode_end(&mut self) {
        self.prev_obs = None;
        self.prev_action = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sarsa_update() {
        let mut agent = Sarsa::new(2, 2, 0.1, 0.9, 0.0, 0);
        let a = agent.act(&0, true);
        let initial = agent.q_values(0)[a];
        agent.handle_step(&0, &a, 1.0, &1, false);
        let _a2 = agent.act(&1, true);
        let updated = agent.q_values(0)[a];
        assert!(updated > initial);
    }
}
