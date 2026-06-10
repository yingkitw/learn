use crate::core::{Environment, Step};
use crate::spaces::{Box, Discrete, Space};
use rand::rngs::StdRng;
use rand::RngCore;
use rand::SeedableRng;

const GRAVITY: f32 = 9.8;
const MASS_CART: f32 = 1.0;
const MASS_POLE: f32 = 0.1;
const TOTAL_MASS: f32 = MASS_CART + MASS_POLE;
const POLE_LENGTH: f32 = 0.5;
const POLEMASS_LENGTH: f32 = MASS_POLE * POLE_LENGTH;
const FORCE_MAG: f32 = 10.0;
const TAU: f32 = 0.02;
const THETA_THRESHOLD: f32 = 12.0 * 2.0 * std::f32::consts::PI / 360.0;
const X_THRESHOLD: f32 = 2.4;

pub struct CartPole {
    state: [f32; 4],
    obs_space: Box,
    act_space: Discrete,
    rng: StdRng,
}

impl CartPole {
    pub fn new(seed: u64) -> Self {
        Self {
            state: [0.0; 4],
            obs_space: Box::new(
                vec![
                    -X_THRESHOLD * 2.0,
                    -f32::INFINITY,
                    -THETA_THRESHOLD * 2.0,
                    -f32::INFINITY,
                ],
                vec![
                    X_THRESHOLD * 2.0,
                    f32::INFINITY,
                    THETA_THRESHOLD * 2.0,
                    f32::INFINITY,
                ],
                vec![4],
            ),
            act_space: Discrete::new(2),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn reset_state(&mut self) {
        for i in 0..4 {
            self.state[i] = self.rng.next_u32() as f32 / u32::MAX as f32 * 0.1 - 0.05;
        }
    }
}

impl Environment for CartPole {
    type Observation = Vec<f32>;
    type Action = usize;

    fn reset(&mut self, seed: Option<u64>) -> Self::Observation {
        if let Some(s) = seed {
            self.rng = StdRng::seed_from_u64(s);
        }
        self.reset_state();
        self.state.to_vec()
    }

    fn step(&mut self, action: &Self::Action) -> Step<Self::Observation> {
        let mut x = self.state[0];
        let mut x_dot = self.state[1];
        let mut theta = self.state[2];
        let mut theta_dot = self.state[3];

        let force = if *action == 1 { FORCE_MAG } else { -FORCE_MAG };
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let temp = (force + POLEMASS_LENGTH * theta_dot * theta_dot * sin_theta) / TOTAL_MASS;
        let thetaacc = (GRAVITY * sin_theta - cos_theta * temp)
            / (POLE_LENGTH * (4.0 / 3.0 - MASS_POLE * cos_theta * cos_theta / TOTAL_MASS));
        let xacc = temp - POLEMASS_LENGTH * thetaacc * cos_theta / TOTAL_MASS;

        x += TAU * x_dot;
        x_dot += TAU * xacc;
        theta += TAU * theta_dot;
        theta_dot += TAU * thetaacc;

        self.state = [x, x_dot, theta, theta_dot];

        let terminated = x < -X_THRESHOLD
            || x > X_THRESHOLD
            || theta < -THETA_THRESHOLD
            || theta > THETA_THRESHOLD;

        Step {
            observation: self.state.to_vec(),
            reward: 1.0,
            terminated,
            truncated: false,
        }
    }

    fn observation_space(&self) -> &dyn Space {
        &self.obs_space
    }

    fn action_space(&self) -> &dyn Space {
        &self.act_space
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartpole() {
        let mut env = CartPole::new(0);
        let obs = env.reset(None);
        assert_eq!(obs.len(), 4);
        let step = env.step(&0);
        assert_eq!(step.observation.len(), 4);
    }
}
