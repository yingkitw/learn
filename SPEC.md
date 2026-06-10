# SPEC — rl_lib

## Purpose
A minimal, from-scratch reinforcement learning library in Rust for educational and prototyping use.

## Core Traits

### Environment
```rust
pub trait Environment {
    type Observation;
    type Action;
    fn reset(&mut self, seed: Option<u64>) -> Self::Observation;
    fn step(&mut self, action: &Self::Action) -> Step<Self::Observation>;
    fn observation_space(&self) -> &dyn Space;
    fn action_space(&self) -> &dyn Space;
}
```

### Agent
```rust
pub trait Agent {
    type Observation;
    type Action;
    fn act(&mut self, obs: &Self::Observation, training: bool) -> Self::Action;
    fn train_step(&mut self, step: &Step<Self::Observation, Self::Action>);
    fn train_episode_end(&mut self);
}
```

### Policy
```rust
pub trait Policy {
    type Observation;
    type Action;
    fn action(&self, obs: &Self::Observation) -> Self::Action;
    fn action_prob(&self, obs: &Self::Observation, action: &Self::Action) -> f32;
}
```

## Spaces

- **Discrete**: finite set of integers `[0, n)`
- **Box**: continuous n-dimensional array with `(low, high)` bounds

## Algorithms

### Q-Learning
- Tabular, epsilon-greedy exploration.
- Update: `Q[s,a] += alpha * (r + gamma * max_a' Q[s',a'] - Q[s,a])`

### SARSA
- Tabular, on-policy.
- Update: `Q[s,a] += alpha * (r + gamma * Q[s',a'] - Q[s,a])`

### REINFORCE
- Policy gradient with Monte-Carlo returns.
- Simple MLP policy network.

### DQN
- Experience replay + target network.
- Simple MLP Q-network.

## Neural Net (Minimal)

- Manual forward/backward for a 2-layer MLP.
- SGD optimizer.
- No external deep-learning crates.

## Environments

- **GridWorld**: 4x4 or 5x5 grid with start, goal, and holes.
- **CartPole**: 1D cart + pole with simple Euler integration.

## Testing

Every module must have unit tests.
Examples must compile and run with `cargo run --example <name>`.
