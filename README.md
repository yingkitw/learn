# rl_lib

A minimal reinforcement learning library written in Rust from scratch.

## Features

- Core abstractions: `Environment`, `Agent`, `Policy`
- Spaces: `Discrete`, `Box` (continuous)
- Algorithms:
  - Q-Learning (tabular)
  - SARSA (tabular)
  - REINFORCE (policy gradient with MLP)
  - DQN (experience replay + target network)
- Included environments:
  - GridWorld
  - CartPole (simple physics)
- Minimal hand-written neural net backprop — no heavy DL frameworks required.

## Quick Start

```bash
cd rl-lib
cargo test
cargo run --example q_learning_gridworld
cargo run --example reinforce_cartpole
```

## Project Structure

- `SPEC.md` — interface and behavior definitions
- `ARCHITECTURE.md` — module layout and design decisions
- `TODO.md` — implementation checklist
