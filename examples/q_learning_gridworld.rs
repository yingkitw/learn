use learn::agents::q_learning::QLearning;
use learn::core::Agent;
use learn::envs::gridworld::GridWorld;
use learn::core::Environment;

fn main() {
    let mut env = GridWorld::new(3, vec![], 42);
    let n_states = env.observation_space().shape()[0];
    let n_actions = env.action_space().shape()[0];
    let mut agent = QLearning::new(n_states, n_actions, 0.2, 0.99, 0.3, 42);

    let n_episodes = 500;
    for ep in 0..n_episodes {
        let mut obs = env.reset(None);
        let mut total_reward = 0.0;
        let mut steps = 0;
        loop {
            let action = agent.act(&obs, true);
            let step = env.step(&action);
            agent.handle_step(&obs, &action, step.reward, &step.observation, step.terminated);
            total_reward += step.reward;
            obs = step.observation;
            steps += 1;
            if step.terminated || step.truncated || steps > 100 {
                agent.episode_end();
                break;
            }
        }
        if ep % 50 == 0 {
            println!("Episode {}: steps = {}, total_reward = {}", ep, steps, total_reward);
        }
    }

    println!("\nLearned policy (argmax Q):");
    let size = (n_states as f32).sqrt() as usize;
    for s in 0..n_states {
        let q = agent.q_values(s);
        let best = q
            .iter()
            .enumerate()
            .max_by(|(_, a): &(usize, &f32), (_, b): &(usize, &f32)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        print!("{:2} ", best);
        if (s + 1) % size == 0 {
            println!();
        }
    }
}
