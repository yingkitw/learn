use rand::RngCore;

#[derive(Clone)]
pub struct Linear {
    pub weights: Vec<f32>, // row-major: output_size x input_size
    pub biases: Vec<f32>,
    input_size: usize,
    output_size: usize,
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize, rng: &mut dyn RngCore) -> Self {
        let scale = (2.0 / input_size as f32).sqrt();
        let mut weights = vec![0.0; input_size * output_size];
        let biases = vec![0.0; output_size];
        for w in weights.iter_mut() {
            *w = (rng.next_u32() as f32 / u32::MAX as f32 * 2.0 - 1.0) * scale;
        }
        Self {
            weights,
            biases,
            input_size,
            output_size,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; self.output_size];
        for j in 0..self.output_size {
            let mut sum = self.biases[j];
            for i in 0..self.input_size {
                sum += input[i] * self.weights[j * self.input_size + i];
            }
            out[j] = sum;
        }
        out
    }

    /// Backward pass given gradient w.r.t. the output (pre-activation).
    /// Returns `(grad_input, grad_weights, grad_biases)`.
    pub fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut grad_input = vec![0.0; self.input_size];
        let mut grad_weights = vec![0.0; self.weights.len()];
        let mut grad_biases = vec![0.0; self.output_size];
        for j in 0..self.output_size {
            let go = grad_output[j];
            grad_biases[j] = go;
            for i in 0..self.input_size {
                grad_weights[j * self.input_size + i] = go * input[i];
                grad_input[i] += go * self.weights[j * self.input_size + i];
            }
        }
        (grad_input, grad_weights, grad_biases)
    }
}

#[derive(Clone)]
pub struct MLP {
    pub layers: Vec<Linear>,
    pre_activations: Vec<Vec<f32>>,
    inputs: Vec<Vec<f32>>,
}

impl MLP {
    pub fn new(sizes: &[usize], rng: &mut dyn RngCore) -> Self {
        let mut layers = Vec::new();
        for i in 0..sizes.len() - 1 {
            layers.push(Linear::new(sizes[i], sizes[i + 1], rng));
        }
        Self {
            layers,
            pre_activations: Vec::new(),
            inputs: Vec::new(),
        }
    }

    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        self.pre_activations.clear();
        self.inputs.clear();
        let mut current = input.to_vec();
        self.inputs.push(current.clone());
        for layer in &self.layers {
            let pre = layer.forward(&current);
            self.pre_activations.push(pre.clone());
            // ReLU on hidden layers
            if self.layers.len() > 1 && self.pre_activations.len() < self.layers.len() {
                current = pre.iter().map(|&x| x.max(0.0)).collect();
            } else {
                current = pre;
            }
            self.inputs.push(current.clone());
        }
        current
    }

    /// Returns weight/bias gradients per layer (same order as `self.layers`).
    pub fn backward(&mut self, grad_output: &[f32]) -> Vec<(Vec<f32>, Vec<f32>)> {
        let mut grad = grad_output.to_vec();
        let mut grads = Vec::new();
        for i in (0..self.layers.len()).rev() {
            if i < self.layers.len() - 1 {
                let pre = &self.pre_activations[i];
                for j in 0..grad.len() {
                    if pre[j] <= 0.0 {
                        grad[j] = 0.0;
                    }
                }
            }
            let (grad_input, gw, gb) = self.layers[i].backward(&self.inputs[i], &grad);
            grads.push((gw, gb));
            grad = grad_input;
        }
        grads.reverse();
        grads
    }

    pub fn update(&mut self, grads: &[(Vec<f32>, Vec<f32>)], lr: f32) {
        for (i, (gw, gb)) in grads.iter().enumerate() {
            for j in 0..self.layers[i].weights.len() {
                self.layers[i].weights[j] -= lr * gw[j];
            }
            for j in 0..self.layers[i].biases.len() {
                self.layers[i].biases[j] -= lr * gb[j];
            }
        }
    }
}

pub fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&v| v / sum).collect()
}

pub fn log_softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let log_sum = sum.ln();
    x.iter().map(|&v| v - max - log_sum).collect()
}

pub fn cross_entropy_grad(logits: &[f32], target: usize) -> Vec<f32> {
    let mut probs = softmax(logits);
    probs[target] -= 1.0;
    probs
}

pub fn mse_grad(pred: &[f32], target: &[f32]) -> Vec<f32> {
    pred.iter()
        .zip(target.iter())
        .map(|(p, t)| 2.0 * (p - t))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_linear_forward_backward() {
        let mut rng = StdRng::seed_from_u64(0);
        let layer = Linear::new(3, 2, &mut rng);
        let input = vec![1.0, 2.0, 3.0];
        let out = layer.forward(&input);
        assert_eq!(out.len(), 2);
        let grad = vec![0.5, -0.5];
        let (g_in, g_w, g_b) = layer.backward(&input, &grad);
        assert_eq!(g_in.len(), 3);
        assert_eq!(g_w.len(), 6);
        assert_eq!(g_b.len(), 2);
    }

    #[test]
    fn test_mlp_forward_backward() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut mlp = MLP::new(&[2, 3, 2], &mut rng);
        let out = mlp.forward(&[1.0, -1.0]);
        assert_eq!(out.len(), 2);
        let grads = mlp.backward(&[1.0, 0.0]);
        assert_eq!(grads.len(), 2);
    }

    #[test]
    fn test_softmax() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
