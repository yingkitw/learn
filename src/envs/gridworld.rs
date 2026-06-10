use crate::core::{Environment, Step};
use crate::spaces::{Discrete, Space};
use rand::rngs::StdRng;
use rand::SeedableRng;

pub struct GridWorld {
    size: usize,
    state: usize,
    holes: Vec<usize>,
    goal: usize,
    obs_space: Discrete,
    act_space: Discrete,
    rng: StdRng,
}

impl GridWorld {
    pub fn new(size: usize, holes: Vec<usize>, seed: u64) -> Self {
        let n = size * size;
        let goal = n - 1;
        Self {
            size,
            state: 0,
            holes,
            goal,
            obs_space: Discrete::new(n),
            act_space: Discrete::new(4),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn default() -> Self {
        Self::new(4, vec![5, 7, 11, 12], 0)
    }

    fn move_agent(&self, state: usize, action: usize) -> usize {
        let row = state / self.size;
        let col = state % self.size;
        let (dr, dc) = match action {
            0 => (-1isize, 0isize), // up
            1 => (0, 1),            // right
            2 => (1, 0),            // down
            3 => (0, -1),           // left
            _ => (0, 0),
        };
        let new_row = (row as isize + dr).clamp(0, self.size as isize - 1) as usize;
        let new_col = (col as isize + dc).clamp(0, self.size as isize - 1) as usize;
        new_row * self.size + new_col
    }
}

impl Environment for GridWorld {
    type Observation = usize;
    type Action = usize;

    fn reset(&mut self, seed: Option<u64>) -> Self::Observation {
        if let Some(s) = seed {
            self.rng = StdRng::seed_from_u64(s);
        }
        self.state = 0;
        self.state
    }

    fn step(&mut self, action: &Self::Action) -> Step<Self::Observation> {
        let next_state = self.move_agent(self.state, *action);
        self.state = next_state;
        let mut reward = -1.0f32;
        let mut terminated = false;
        if self.state == self.goal {
            reward = 10.0;
            terminated = true;
        } else if self.holes.contains(&self.state) {
            reward = -10.0;
            terminated = true;
        }
        Step {
            observation: self.state,
            reward,
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
    fn test_gridworld() {
        let mut env = GridWorld::default();
        let s = env.reset(None);
        assert_eq!(s, 0);
        let step = env.step(&1);
        assert_eq!(step.observation, 1);
        assert_eq!(step.reward, -1.0);
    }
}
