use crate::core::Agent;
use crate::nn::{cross_entropy_grad, softmax, MLP};
use crate::utils::compute_returns;
use rand::rngs::StdRng;
use rand::RngCore;
use rand::SeedableRng;

pub struct Reinforce {
    policy: MLP,
    lr: f32,
    gamma: f32,
    rng: StdRng,
    observations: Vec<Vec<f32>>,
    actions: Vec<usize>,
    rewards: Vec<f32>,
}

impl Reinforce {
    pub fn new(policy: MLP, lr: f32, gamma: f32, seed: u64) -> Self {
        Self {
            policy,
            lr,
            gamma,
            rng: StdRng::seed_from_u64(seed),
            observations: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
        }
    }
}

impl Agent for Reinforce {
    type Observation = Vec<f32>;
    type Action = usize;

    fn act(&mut self, obs: &Self::Observation, training: bool) -> Self::Action {
        let logits = self.policy.forward(obs);
        if training {
            let probs = softmax(&logits);
            let r = self.rng.next_u32() as f32 / u32::MAX as f32;
            let mut cumsum = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r < cumsum {
                    return i;
                }
            }
            probs.len() - 1
        } else {
            logits
                .iter()
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
        _next_obs: &Self::Observation,
        _done: bool,
    ) {
        self.observations.push(obs.clone());
        self.actions.push(*action);
        self.rewards.push(reward);
    }

    fn episode_end(&mut self) {
        if self.observations.is_empty() {
            return;
        }
        let returns = compute_returns(&self.rewards, self.gamma);
        let mut total_grads: Vec<(Vec<f32>, Vec<f32>)> = self
            .policy
            .layers
            .iter()
            .map(|l| (vec![0.0; l.weights.len()], vec![0.0; l.biases.len()]))
            .collect();

        for t in 0..self.observations.len() {
            let logits = self.policy.forward(&self.observations[t]);
            let grad = cross_entropy_grad(&logits, self.actions[t]);
            let scaled_grad: Vec<f32> = grad.iter().map(|&g| g * returns[t]).collect();
            let grads = self.policy.backward(&scaled_grad);
            for (i, (gw, gb)) in grads.iter().enumerate() {
                for j in 0..total_grads[i].0.len() {
                    total_grads[i].0[j] += gw[j];
                }
                for j in 0..total_grads[i].1.len() {
                    total_grads[i].1[j] += gb[j];
                }
            }
        }

        let n = self.observations.len() as f32;
        for (gw, gb) in total_grads.iter_mut() {
            for k in gw.iter_mut() {
                *k /= n;
            }
            for k in gb.iter_mut() {
                *k /= n;
            }
        }

        self.policy.update(&total_grads, self.lr);
        self.observations.clear();
        self.actions.clear();
        self.rewards.clear();
    }
}
