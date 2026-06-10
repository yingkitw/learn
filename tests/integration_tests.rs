use learn::agents::dqn::Dqn;
use learn::agents::q_learning::QLearning;
use learn::agents::reinforce::Reinforce;
use learn::agents::sarsa::Sarsa;
use learn::core::Agent;
use learn::core::Environment;
use learn::envs::cartpole::CartPole;
use learn::envs::gridworld::GridWorld;
use learn::nn::MLP;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn test_q_learning_solves_gridworld() {
    let mut env = GridWorld::new(3, vec![], 0);
    let n_states = env.observation_space().shape()[0];
    let n_actions = env.action_space().shape()[0];
    let mut agent = QLearning::new(n_states, n_actions, 0.5, 0.9, 0.3, 0);

    for _ in 0..300 {
        let mut obs = env.reset(None);
        let mut steps = 0;
        loop {
            let action = agent.act(&obs, true);
            let step = env.step(&action);
            agent.handle_step(&obs, &action, step.reward, &step.observation, step.terminated);
            obs = step.observation;
            steps += 1;
            if step.terminated || steps > 50 {
                agent.episode_end();
                break;
            }
        }
    }

    // Evaluate: should reach goal in <= 10 steps consistently
    let mut total_steps = 0;
    for _ in 0..10 {
        let mut obs = env.reset(None);
        let mut steps = 0;
        loop {
            let action = agent.act(&obs, false);
            let step = env.step(&action);
            obs = step.observation;
            steps += 1;
            if step.terminated || steps > 20 {
                break;
            }
        }
        total_steps += steps;
    }
    let avg_steps = total_steps as f32 / 10.0;
    assert!(
        avg_steps <= 10.0,
        "QLearning did not converge: avg_steps = {}",
        avg_steps
    );
}

#[test]
fn test_sarsa_solves_gridworld() {
    let mut env = GridWorld::new(3, vec![], 0);
    let n_states = env.observation_space().shape()[0];
    let n_actions = env.action_space().shape()[0];
    let mut agent = Sarsa::new(n_states, n_actions, 0.5, 0.9, 0.3, 0);

    for _ in 0..300 {
        let mut obs = env.reset(None);
        let mut steps = 0;
        loop {
            let action = agent.act(&obs, true);
            let step = env.step(&action);
            agent.handle_step(&obs, &action, step.reward, &step.observation, step.terminated);
            obs = step.observation;
            steps += 1;
            if step.terminated || steps > 50 {
                agent.episode_end();
                break;
            }
        }
    }

    let mut total_steps = 0;
    for _ in 0..10 {
        let mut obs = env.reset(None);
        let mut steps = 0;
        loop {
            let action = agent.act(&obs, false);
            let step = env.step(&action);
            obs = step.observation;
            steps += 1;
            if step.terminated || steps > 20 {
                break;
            }
        }
        total_steps += steps;
    }
    let avg_steps = total_steps as f32 / 10.0;
    assert!(
        avg_steps <= 10.0,
        "SARSA did not converge: avg_steps = {}",
        avg_steps
    );
}

#[test]
fn test_reinforce_improves_cartpole() {
    let mut rng = StdRng::seed_from_u64(0);
    let policy = MLP::new(&[4, 16, 2], &mut rng);
    let mut agent = Reinforce::new(policy, 0.01, 0.99, 0);
    let mut env = CartPole::new(0);

    let mut rewards = Vec::new();
    for _ in 0..100 {
        let mut obs = env.reset(None);
        let mut total_reward = 0.0;
        let mut steps = 0;
        loop {
            let action = agent.act(&obs, true);
            let step = env.step(&action);
            agent.handle_step(
                &obs,
                &action,
                step.reward,
                &step.observation,
                step.terminated,
            );
            total_reward += step.reward;
            obs = step.observation;
            steps += 1;
            if step.terminated || steps > 200 {
                agent.episode_end();
                break;
            }
        }
        rewards.push(total_reward);
    }

    let first_10_avg: f32 = rewards.iter().take(10).sum::<f32>() / 10.0;
    let last_10_avg: f32 = rewards.iter().skip(90).sum::<f32>() / 10.0;
    assert!(
        last_10_avg >= first_10_avg,
        "REINFORCE did not improve: first_10_avg = {}, last_10_avg = {}",
        first_10_avg,
        last_10_avg
    );
}

#[test]
fn test_dqn_learns_cartpole() {
    let mut rng = StdRng::seed_from_u64(0);
    let q_network = MLP::new(&[4, 32, 2], &mut rng);
    let mut agent = Dqn::new(q_network, 5000, 0.2, 0.99, 0.01, 16, 50, 2, 0);
    let mut env = CartPole::new(0);

    let mut rewards = Vec::new();
    for _ in 0..200 {
        let mut obs = env.reset(None);
        let mut total_reward = 0.0;
        let mut steps = 0;
        loop {
            let action = agent.act(&obs, true);
            let step = env.step(&action);
            agent.handle_step(
                &obs,
                &action,
                step.reward,
                &step.observation,
                step.terminated,
            );
            total_reward += step.reward;
            obs = step.observation;
            steps += 1;
            if step.terminated || steps > 200 {
                agent.episode_end();
                break;
            }
        }
        rewards.push(total_reward);
    }

    let first_20_avg: f32 = rewards.iter().take(20).sum::<f32>() / 20.0;
    let last_20_avg: f32 = rewards.iter().skip(180).sum::<f32>() / 20.0;
    assert!(
        last_20_avg >= first_20_avg,
        "DQN did not improve: first_20_avg = {}, last_20_avg = {}",
        first_20_avg,
        last_20_avg
    );
}

#[test]
fn test_gridworld_hole_termination() {
    let mut env = GridWorld::new(4, vec![5], 0);
    env.reset(None);
    // Move to state 5 via path: 0 -> 1 -> 5 (right, down)
    let _ = env.step(&1); // right to 1
    let step = env.step(&2); // down to 5 (hole)
    assert!(step.terminated);
    assert_eq!(step.reward, -10.0);
}

#[test]
fn test_cartpole_termination_on_angle() {
    let mut env = CartPole::new(0);
    env.reset(Some(0));
    // Apply force repeatedly in one direction to make pole fall
    let mut terminated = false;
    for _ in 0..200 {
        let step = env.step(&1);
        if step.terminated {
            terminated = true;
            break;
        }
    }
    assert!(terminated, "CartPole should terminate when pole falls");
}

#[test]
fn test_replay_buffer_capacity() {
    use learn::buffer::ReplayBuffer;
    use learn::core::Transition;

    let mut buf: ReplayBuffer<f32, usize> = ReplayBuffer::new(3);
    for i in 0..5 {
        buf.add(Transition {
            obs: i as f32,
            action: i,
            reward: i as f32,
            next_obs: (i + 1) as f32,
            terminated: false,
            truncated: false,
        });
    }
    assert_eq!(buf.len(), 3);
}

#[test]
fn test_nn_mlp_update_changes_weights() {
    use learn::nn::MLP;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let mut rng = StdRng::seed_from_u64(0);
    let mut mlp = MLP::new(&[2, 3, 2], &mut rng);
    let initial_w = mlp.layers[0].weights[0];

    let out = mlp.forward(&[1.0, 0.5]);
    let target = vec![0.0, 1.0];
    let grad: Vec<f32> = out.iter().zip(target.iter()).map(|(p, t)| 2.0 * (p - t)).collect();
    let grads = mlp.backward(&grad);
    mlp.update(&grads, 0.1);

    let updated_w = mlp.layers[0].weights[0];
    assert_ne!(initial_w, updated_w, "Weights should change after update");
}
