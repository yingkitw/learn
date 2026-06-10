use crate::spaces::Space;

#[derive(Debug, Clone)]
pub struct Step<O> {
    pub observation: O,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
}

#[derive(Debug, Clone)]
pub struct Transition<O, A> {
    pub obs: O,
    pub action: A,
    pub reward: f32,
    pub next_obs: O,
    pub terminated: bool,
    pub truncated: bool,
}

pub trait Environment {
    type Observation;
    type Action;
    fn reset(&mut self, seed: Option<u64>) -> Self::Observation;
    fn step(&mut self, action: &Self::Action) -> Step<Self::Observation>;
    fn observation_space(&self) -> &dyn Space;
    fn action_space(&self) -> &dyn Space;
}

pub trait Agent {
    type Observation;
    type Action;
    fn act(&mut self, obs: &Self::Observation, training: bool) -> Self::Action;
    fn handle_step(
        &mut self,
        obs: &Self::Observation,
        action: &Self::Action,
        reward: f32,
        next_obs: &Self::Observation,
        done: bool,
    );
    fn episode_end(&mut self);
}

pub trait Policy {
    type Observation;
    type Action;
    fn action(&self, obs: &Self::Observation) -> Self::Action;
    fn action_prob(&self, obs: &Self::Observation, action: &Self::Action) -> f32;
}
