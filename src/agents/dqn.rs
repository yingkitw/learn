use crate::buffer::ReplayBuffer;
use crate::core::{Agent, Transition};
use crate::nn::{mse_grad, MLP};
use rand::rngs::StdRng;
use rand::RngCore;
use rand::SeedableRng;

pub struct Dqn {
    q_network: MLP,
    target_network: MLP,
    buffer: ReplayBuffer<Vec<f32>, usize>,
    epsilon: f32,
    gamma: f32,
    lr: f32,
    batch_size: usize,
    update_every: usize,
    step_count: usize,
    n_actions: usize,
    rng: StdRng,
}

impl Dqn {
    pub fn new(
        q_network: MLP,
        buffer_capacity: usize,
        epsilon: f32,
        gamma: f32,
        lr: f32,
        batch_size: usize,
        update_every: usize,
        n_actions: usize,
        seed: u64,
    ) -> Self {
        let target_network = q_network.clone();
        Self {
            q_network,
            target_network,
            buffer: ReplayBuffer::new(buffer_capacity),
            epsilon,
            gamma,
            lr,
            batch_size,
            update_every,
            step_count: 0,
            n_actions,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Agent for Dqn {
    type Observation = Vec<f32>;
    type Action = usize;

    fn act(&mut self, obs: &Self::Observation, training: bool) -> Self::Action {
        let q_values = self.q_network.forward(obs);
        if training && (self.rng.next_u32() as f32 / u32::MAX as f32) < self.epsilon {
            (self.rng.next_u32() as usize) % self.n_actions
        } else {
            q_values
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
        next_obs: &Self::Observation,
        done: bool,
    ) {
        self.buffer.add(Transition {
            obs: obs.clone(),
            action: *action,
            reward,
            next_obs: next_obs.clone(),
            terminated: done,
            truncated: false,
        });
        self.step_count += 1;

        if self.buffer.len() >= self.batch_size {
            if let Some(batch) = self.buffer.sample(self.batch_size, &mut self.rng) {
                let mut targets = Vec::with_capacity(batch.len());
                for t in &batch {
                    let next_q = self.target_network.forward(&t.next_obs);
                    let max_next = if t.terminated {
                        0.0
                    } else {
                        next_q
                            .iter()
                            .cloned()
                            .fold(f32::NEG_INFINITY, f32::max)
                    };
                    targets.push(t.reward + self.gamma * max_next);
                }

                let mut total_grads: Vec<(Vec<f32>, Vec<f32>)> = self
                    .q_network
                    .layers
                    .iter()
                    .map(|l| (vec![0.0; l.weights.len()], vec![0.0; l.biases.len()]))
                    .collect();

                for (i, t) in batch.iter().enumerate() {
                    let q_values = self.q_network.forward(&t.obs);
                    let mut target_vec = q_values.clone();
                    target_vec[t.action] = targets[i];
                    let grad = mse_grad(&q_values, &target_vec);
                    let grads = self.q_network.backward(&grad);
                    for (j, (gw, gb)) in grads.iter().enumerate() {
                        for k in 0..total_grads[j].0.len() {
                            total_grads[j].0[k] += gw[k];
                        }
                        for k in 0..total_grads[j].1.len() {
                            total_grads[j].1[k] += gb[k];
                        }
                    }
                }

                let n = batch.len() as f32;
                for (gw, gb) in total_grads.iter_mut() {
                    for k in gw.iter_mut() {
                        *k /= n;
                    }
                    for k in gb.iter_mut() {
                        *k /= n;
                    }
                }

                self.q_network.update(&total_grads, self.lr);
            }
        }

        if self.step_count % self.update_every == 0 {
            for i in 0..self.q_network.layers.len() {
                self.target_network.layers[i]
                    .weights
                    .clone_from(&self.q_network.layers[i].weights);
                self.target_network.layers[i]
                    .biases
                    .clone_from(&self.q_network.layers[i].biases);
            }
        }
    }

    fn episode_end(&mut self) {}
}
