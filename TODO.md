# TODO

## Phase 1: Foundations

- [x] Project scaffolding (`cargo init --lib`)
- [x] README, SPEC, ARCHITECTURE
- [x] Core traits (`src/core.rs`)
  - [x] `Environment` trait
  - [x] `Agent` trait
  - [x] `Policy` trait
- [x] Spaces (`src/spaces.rs`)
  - [x] `Space` trait
  - [x] `Discrete` space
  - [x] `Box` (continuous) space

## Phase 2: Utilities & Data Structures

- [x] Replay Buffer (`src/buffer.rs`)
  - [x] `ReplayBuffer` struct with sample & add
- [x] Utilities (`src/utils.rs`)
  - [x] `epsilon_greedy`
  - [x] `compute_returns` (discounted returns)
  - [x] `one_hot` encode

## Phase 3: Tabular Algorithms

- [x] Q-Learning (`src/agents/q_learning.rs`)
- [x] SARSA (`src/agents/sarsa.rs`)

## Phase 4: Function Approximation

- [x] Simple neural net module (`src/nn.rs`)
  - [x] `Layer` (linear + ReLU)
  - [x] `MLP`
  - [x] Forward pass
  - [x] Backward pass / SGD update
- [x] REINFORCE (`src/agents/reinforce.rs`)
- [x] DQN (`src/agents/dqn.rs`)

## Phase 5: Environments & Examples

- [x] GridWorld environment (`src/envs/gridworld.rs`)
- [x] CartPole-like environment (`src/envs/cartpole.rs`)
- [x] Examples (`examples/`)
  - [x] `q_learning_gridworld.rs`
  - [x] `reinforce_cartpole.rs`

## Phase 6: Polish

- [x] Unit tests for all modules
- [x] Cargo clippy / fmt clean
- [x] Documentation comments
