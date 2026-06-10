# ARCHITECTURE — rl_lib

## Module Layout

```
src/
  lib.rs          — re-exports
  core.rs         — Environment, Agent, Policy, Step
  spaces.rs       — Space, Discrete, Box
  buffer.rs       — ReplayBuffer
  utils.rs        — helpers (epsilon_greedy, returns, one_hot)
  nn.rs           — minimal tensor + MLP (manual backprop)
  agents/
    mod.rs
    q_learning.rs
    sarsa.rs
    reinforce.rs
    dqn.rs
  envs/
    mod.rs
    gridworld.rs
    cartpole.rs
examples/
  q_learning_gridworld.rs
  reinforce_cartpole.rs
```

## Design Decisions

- **No external ML crates**: Only `rand` for RNG. All linear algebra and backprop are hand-written to keep the library self-contained and educational.
- **Generic over Observation/Action**: Core traits use associated types so users can plug in their own state/action representations.
- **Domain modules**: `agents/`, `envs/`, `nn/` are domain-driven rather than layer-driven.
- **Manual backprop**: The `nn.rs` module keeps a simple tape-less approach — store weights, biases, and compute gradients analytically for a fully-connected MLP.
