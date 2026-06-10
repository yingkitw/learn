use learn::agents::reinforce::Reinforce;
use learn::core::Agent;
use learn::core::Environment;
use learn::envs::cartpole::CartPole;
use learn::nn::MLP;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let policy = MLP::new(&[4, 16, 2], &mut rng);
    let mut agent = Reinforce::new(policy, 0.01, 0.99, 42);
    let mut env = CartPole::new(42);

    let n_episodes = 2000;
    let mut last_100: Vec<f32> = Vec::with_capacity(100);

    for ep in 0..n_episodes {
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
            if step.terminated || step.truncated || steps > 500 {
                agent.episode_end();
                break;
            }
        }

        last_100.push(total_reward);
        if last_100.len() > 100 {
            last_100.remove(0);
        }

        if ep % 100 == 0 {
            let avg: f32 = last_100.iter().sum::<f32>() / last_100.len().max(1) as f32;
            println!(
                "Episode {}: steps = {}, reward = {}, avg_100 = {:.1}",
                ep, steps, total_reward, avg
            );
        }
    }
}
