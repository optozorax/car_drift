use crate::common::pairs;
use rand::prelude::SliceRandom;
use rand::Rng;

pub fn mul_matrix(
    input: &[f32],
    matrix: &[f32],
    output: &mut [f32],
    input_size: usize,
    output_size: usize,
) {
    for i in 0..output_size {
        output[i] = input
            .iter()
            .zip(&matrix[i * input_size..(i + 1) * input_size])
            .map(|(&a, &b)| a * b)
            .sum();
    }
}

fn sigmoid(x: f32) -> f32 {
    1_f32 / (1_f32 + std::f32::consts::E.powf(-x))
}

fn relu1(x: f32) -> f32 {
    x.clamp(0., 10.)
}

fn relu2(x: f32) -> f32 {
    if x > 0. {
        x
    } else {
        x * 0.1
    }.clamp(-10., 10.)
}

fn activation(x: f32) -> f32 {
    // sigmoid(x)
    relu1(x)
    // relu2(x)
}

fn activation_vector(output: &mut [f32]) {
    for x in output {
        *x = activation(*x);
    }
}

fn sum_vectors(input: &[f32], vector: &[f32], output: &mut [f32]) {
    for (out, (in1, in2)) in output.iter_mut().zip(input.iter().zip(vector.iter())) {
        *out = in1 + in2;
    }
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct NeuralNetwork {
    sizes: Vec<usize>,
    values: Vec<f32>,

    reserved1: Vec<f32>,
    reserved2: Vec<f32>,
}

impl NeuralNetwork {
    pub fn new(sizes: Vec<usize>) -> Self {
        let values_size = pairs(sizes.iter()).map(|(a, b)| (a + 1) * b).sum();
        let mut result = Self {
            sizes,
            values: vec![0.; values_size],
            reserved1: Default::default(),
            reserved2: Default::default(),
        };
        result.resize_reserved();
        result
    }

    pub fn get_sizes(&self) -> &[usize] {
        &self.sizes
    }

    pub fn get_values(&self) -> &[f32] {
        &self.values
    }

    pub fn get_values_mut(&mut self) -> &mut [f32] {
        &mut self.values
    }

    pub fn generate_random(sizes: Vec<usize>, rng: &mut impl Rng) -> Self {
        let mut result = Self::new(sizes);
        result
            .values
            .iter_mut()
            .for_each(|x| *x = rng.gen_range(-1.0..1.0));
        result
    }

    pub fn resize_reserved(&mut self) {
        let reserved_size = pairs(self.sizes.iter())
            .map(|(a, b)| (a + b).max(a + a).max(b + b))
            .max()
            .unwrap_or(0);
        self.reserved1 = vec![0.; reserved_size];
        self.reserved2 = vec![0.; reserved_size];
    }

    pub fn calc<'a>(&'a mut self, input: &[f32]) -> &'a [f32] {
        input
            .iter()
            .enumerate()
            .for_each(|(i, x)| self.reserved1[i] = *x);

        let mut offset = 0;
        for (prev, now) in pairs(self.sizes.iter()) {
            let (prev_slice, now_slice) = self.reserved1[..prev + now].split_at_mut(*prev);
            mul_matrix(
                prev_slice,
                &self.values[offset..(offset + prev * now)],
                now_slice,
                *prev,
                *now,
            );
            offset += prev * now;
            sum_vectors(
                now_slice,
                &self.values[offset..(offset + now)],
                &mut self.reserved2[..*now],
            );
            activation_vector(&mut self.reserved2[..*now]);
            offset += now;
            std::mem::swap(&mut self.reserved1, &mut self.reserved2);
        }

        &self.reserved1[..*self.sizes.last().unwrap()]
    }

    pub fn mutate_float_value(&mut self, rng: &mut impl Rng) {
        *self.values.choose_mut(rng).unwrap() += rng.gen_range(-0.1..0.1);
    }
}

#[derive(Clone, Debug)]
pub struct Layer {
    input_size: usize,
    output_size: usize,
    matrix: Vec<Vec<f32>>, // outer vec has size output_size, inner vec has size input_size
    bias: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct NeuralNetworkUnoptimized {
    layers: Vec<Layer>,
}

impl NeuralNetworkUnoptimized {
    pub fn calc(&self, input: &[f32]) -> Vec<f32> {
        self.layers.iter().fold(input.to_vec(), |input, layer| {
            let mut output = vec![0.0; layer.output_size];
            for (i, output_neuron) in output.iter_mut().enumerate() {
                *output_neuron = activation(
                    layer.matrix[i]
                        .iter()
                        .zip(input.iter())
                        .map(|(&weight, &input)| weight * input)
                        .sum::<f32>()
                        + layer.bias[i],
                );
            }
            output
        })
    }
}

impl NeuralNetwork {
    pub fn to_unoptimized(&self) -> NeuralNetworkUnoptimized {
        let mut offset = 0;
        let layers = pairs(self.sizes.iter())
            .map(|(&input_size, &output_size)| {
                let matrix_size = input_size * output_size;
                let matrix = self.values[offset..offset + matrix_size]
                    .chunks(input_size)
                    .map(|row| row.to_vec())
                    .collect();
                offset += matrix_size;
                let bias = self.values[offset..offset + output_size].to_vec();
                offset += output_size;
                Layer {
                    input_size,
                    output_size,
                    matrix,
                    bias,
                }
            })
            .collect();

        NeuralNetworkUnoptimized { layers }
    }

    pub fn from_unoptimized(unoptimized: &NeuralNetworkUnoptimized) -> Self {
        let sizes: Vec<usize> = std::iter::once(unoptimized.layers[0].input_size)
            .chain(unoptimized.layers.iter().map(|layer| layer.output_size))
            .collect();

        let values: Vec<f32> = unoptimized
            .layers
            .iter()
            .flat_map(|layer| {
                layer
                    .matrix
                    .iter()
                    .flatten()
                    .chain(layer.bias.iter())
                    .cloned()
            })
            .collect();

        let mut nn = Self {
            sizes,
            values,
            reserved1: Default::default(),
            reserved2: Default::default(),
        };
        nn.resize_reserved();
        nn
    }
}

impl NeuralNetworkUnoptimized {
    pub fn to_optimized(&self) -> NeuralNetwork {
        NeuralNetwork::from_unoptimized(self)
    }

    pub fn from_optimized(optimized: &NeuralNetwork) -> Self {
        optimized.to_unoptimized()
    }
}

impl NeuralNetworkUnoptimized {
    pub fn mutate_float_value(&mut self, rng: &mut impl Rng) {
        let layer = self.layers.choose_mut(rng).unwrap();
        let row = rng.gen_range(0..layer.matrix.len());
        let col = rng.gen_range(0..layer.matrix[row].len());
        layer.matrix[row][col] += rng.gen_range(-0.1..0.1);
    }

    pub fn add_random_hidden_neuron(&mut self, rng: &mut impl Rng) {
        if self.layers.len() <= 1 {
            return;
        }
        let layer_index = rng.gen_range(0..self.layers.len() - 1);
        self.add_hidden_neuron(layer_index);
    }

    // layer_index = 0 - first layer
    // layer_index = layers.size() - 1 - last layer
    pub fn add_hidden_neuron(&mut self, layer_index: usize) {
        if self.layers.len() <= 1 {
            return;
        }
        let layer = &mut self.layers[layer_index];
        layer.output_size += 1;
        layer.matrix.push(vec![0.0; layer.input_size]);
        layer.bias.push(0.0);

        if layer_index < self.layers.len() - 1 {
            let next_layer = &mut self.layers[layer_index + 1];
            next_layer.input_size += 1;
            for row in &mut next_layer.matrix {
                row.push(0.0);
            }
        }
    }

    
    pub fn add_random_hidden_layer(&mut self, rng: &mut impl Rng) {
        let layer_index = rng.gen_range(0..=self.layers.len());
        self.add_hidden_layer(layer_index);
    }

    // layer_index = 0 - layer right after input
    // layer_index = layers.len() - layer right before output
    pub fn add_hidden_layer(&mut self, layer_index: usize) {
        let input_size = if layer_index == 0 {
            self.layers[0].input_size
        } else {
            self.layers[layer_index - 1].output_size
        };
        let output_size = if layer_index == self.layers.len() {
            self.layers.last().unwrap().output_size
        } else {
            self.layers[layer_index].input_size
        };

        let mut matrix = vec![vec![0.0; input_size]; output_size];
        #[allow(clippy::needless_range_loop)]
        for i in 0..input_size.min(output_size) {
            matrix[i][i] = 1.0;
        }

        let new_layer = Layer {
            input_size,
            output_size,
            matrix,
            bias: vec![0.0; output_size],
        };

        self.layers.insert(layer_index, new_layer);
    }

    pub fn remove_hidden_neuron(&mut self, rng: &mut impl Rng) {
        if self.layers.len() <= 1 || self.layers.iter().any(|l| l.output_size <= 1) {
            return;
        }

        let layer_index = rng.gen_range(0..self.layers.len() - 1);
        let layer = &mut self.layers[layer_index];
        let neuron_index = rng.gen_range(0..layer.output_size);

        layer.output_size -= 1;
        layer.matrix.remove(neuron_index);
        layer.bias.remove(neuron_index);

        let next_layer = &mut self.layers[layer_index + 1];
        next_layer.input_size -= 1;
        for row in &mut next_layer.matrix {
            row.remove(neuron_index);
        }
    }

    pub fn remove_hidden_layer(&mut self, rng: &mut impl Rng) {
        if self.layers.len() <= 1 {
            return;
        }

        let layer_index = rng.gen_range(0..self.layers.len());

        if layer_index == self.layers.len() - 1 {
            // Removing the last layer
            let removed_layer = self.layers.pop().unwrap();
            let prev_layer = self.layers.last_mut().unwrap();
            prev_layer.output_size = removed_layer.output_size;
            prev_layer
                .matrix
                .resize(removed_layer.output_size, vec![0.0; prev_layer.input_size]);
            prev_layer.bias.resize(removed_layer.output_size, 0.0);
        } else {
            // Removing a hidden layer
            let removed_layer = self.layers.remove(layer_index);
            let next_layer = &mut self.layers[layer_index];

            // Adjust the input size of the next layer
            next_layer.input_size = removed_layer.input_size;

            // Resize the weight matrix of the next layer
            for row in &mut next_layer.matrix {
                row.resize(next_layer.input_size, 0.0);
            }
        }
    }

    pub fn generate_neuron_counts(rng: &mut impl Rng, size: usize) -> Vec<usize> {
        (0..size).map(|_| rng.gen_range(1..10)).collect()
    }

    pub fn generate_random(rng: &mut impl Rng, neuron_counts: &[usize]) -> Self {
        let mut layers = Vec::new();
        for i in 0..neuron_counts.len() - 1 {
            let input_size = neuron_counts[i];
            let output_size = neuron_counts[i + 1];
            let matrix = (0..output_size)
                .map(|_| (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
                .collect();
            let bias = (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
            layers.push(Layer {
                input_size,
                output_size,
                matrix,
                bias,
            });
        }
        NeuralNetworkUnoptimized { layers }
    }
}

#[cfg(test)]
mod tests2 {
    use super::*;
    use approx::assert_relative_eq;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn generate_random_input(rng: &mut impl Rng, size: usize) -> Vec<f32> {
        (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    #[test]
    fn test_add_neuron_output_unchanged() {
        let mut rng = StdRng::seed_from_u64(42);
        for size in 2..10 {
            for _ in 0..10 {
                let neuron_counts =
                    NeuralNetworkUnoptimized::generate_neuron_counts(&mut rng, size);
                let mut nn = NeuralNetworkUnoptimized::generate_random(&mut rng, &neuron_counts);
                let input = generate_random_input(&mut rng, neuron_counts[0]);
                let output_before = nn.calc(&input);
                nn.add_random_hidden_neuron(&mut rng);
                let output_after = nn.calc(&input);
                assert_relative_eq!(
                    output_before.as_slice(),
                    output_after.as_slice(),
                    epsilon = 1e-5
                );
            }
        }
    }

    #[test]
    fn test_add_layer_output_unchanged() {
        let mut rng = StdRng::seed_from_u64(42);
        for size in 2..10 {
            for _ in 0..10 {
                let neuron_counts =
                    NeuralNetworkUnoptimized::generate_neuron_counts(&mut rng, size);
                let mut nn = NeuralNetworkUnoptimized::generate_random(&mut rng, &neuron_counts);
                let mut input = generate_random_input(&mut rng, neuron_counts[0]);
                activation_vector(&mut input);
                let output_before = nn.calc(&input);
                nn.add_random_hidden_layer(&mut rng);
                let output_after = nn.calc(&input);
                assert_relative_eq!(
                    output_before.as_slice(),
                    output_after.as_slice(),
                    epsilon = 1e-5
                );
            }
        }
    }

    #[test]
    fn test_remove_neuron_valid_structure() {
        let mut rng = StdRng::seed_from_u64(42);
        for size in 2..10 {
            for _ in 0..10 {
                let neuron_counts =
                    NeuralNetworkUnoptimized::generate_neuron_counts(&mut rng, size);
                let mut nn = NeuralNetworkUnoptimized::generate_random(&mut rng, &neuron_counts);
                let input_size_before = nn.layers[0].input_size;
                let output_size_before = nn.layers.last().unwrap().output_size;
                nn.remove_hidden_neuron(&mut rng);
                assert_eq!(nn.layers[0].input_size, input_size_before);
                assert_eq!(nn.layers.last().unwrap().output_size, output_size_before);
                assert!(nn.layers.iter().all(|l| l.output_size > 0));

                // Check that all layers have matching input/output sizes
                for i in 1..nn.layers.len() {
                    assert_eq!(nn.layers[i - 1].output_size, nn.layers[i].input_size);
                }
            }
        }
    }

    #[test]
    fn test_remove_layer_valid_structure() {
        let mut rng = StdRng::seed_from_u64(42);
        for size in 2..10 {
            for _ in 0..10 {
                let neuron_counts =
                    NeuralNetworkUnoptimized::generate_neuron_counts(&mut rng, size);
                let mut nn = NeuralNetworkUnoptimized::generate_random(&mut rng, &neuron_counts);
                let input_size_before = nn.layers[0].input_size;
                let output_size_before = nn.layers.last().unwrap().output_size;
                let layers_before = nn.layers.len();
                nn.remove_hidden_layer(&mut rng);
                assert_eq!(nn.layers[0].input_size, input_size_before);
                assert_eq!(nn.layers.last().unwrap().output_size, output_size_before);
                if layers_before != 1 {
                    assert_eq!(nn.layers.len(), layers_before - 1);
                }
                assert!(!nn.layers.is_empty());

                // Check that all layers have matching input/output sizes
                for i in 1..nn.layers.len() {
                    assert_eq!(nn.layers[i - 1].output_size, nn.layers[i].input_size);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_network_3_2_3() {
        let mut nn = NeuralNetwork::new(vec![3, 2, 3]);

        // Set weights and biases manually
        nn.values = vec![
            // First layer (3x2 matrix transposed + 2 biases)
            1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 0.1, 0.2,
            // Second layer (2x3 matrix transposed + 3 biases)
            1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 0.3, 0.4, 0.5,
        ];

        let input = vec![0.5, -0.5, 1.0];
        let output = nn.calc(&input);

        // Manual calculation:
        // First layer:
        // [0.5, -0.5, 1.0] * [1 2; 3 4; 5 6] + [0.1, 0.2] = [4.1, 5.2]
        // Second layer:
        // [4.1, 5.2] * [1 2 3; 4 5 6] + [0.3, 0.4, 0.5] = [25.2, 34.6, 44.0]

        assert_eq!(output.len(), 3);
        assert!((output[0] - 25.2).abs() < 1e-3);
        assert!((output[1] - 34.6).abs() < 1e-3);
        assert!((output[2] - 44.0).abs() < 1e-3);
    }

    #[test]
    fn test_neural_network_2_2_1() {
        let mut nn = NeuralNetwork::new(vec![2, 2, 1]);

        nn.values = vec![
            // First layer (2x2 matrix transposed + 2 biases)
            0.5, 1.5, 1.0, 2.0, 0.1, 0.1, // Second layer (2x1 matrix transposed + 1 bias)
            1.0, 2.0, 0.3,
        ];

        let input = vec![1.0, -1.0];
        let output = nn.calc(&input);

        // Manual calculation:
        // First layer:
        // [1.0, -1.0] * [0.5 1.0; 1.5 2.0] + [0.1, 0.1] = [-0.9, -0.9] -> [0., 0.]
        // Second layer:
        // [0., 0.] * [1; 2] + [0.3] = 0.3

        assert_eq!(output.len(), 1);
        assert!((output[0] - (0.3)).abs() < 1e-6);
    }

    fn create_test_network() -> NeuralNetwork {
        let sizes = vec![3, 4, 2];
        let mut nn = NeuralNetwork::new(sizes);
        nn.values = vec![
            // First layer (3x4 matrix + 4 biases)
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 0.01, 0.02, 0.03, 0.04,
            // Second layer (4x2 matrix + 2 biases)
            1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 0.05, 0.06,
        ];
        nn.resize_reserved();
        nn
    }

    #[test]
    fn test_conversion_roundtrip() {
        let original = create_test_network();
        let unoptimized = original.to_unoptimized();
        let roundtrip = NeuralNetwork::from_unoptimized(&unoptimized);

        assert_eq!(original.sizes, roundtrip.sizes);
        assert_eq!(original.values, roundtrip.values);
    }

    #[test]
    fn test_calc_equivalence() {
        let mut original = create_test_network();
        let unoptimized = original.to_unoptimized();

        let input = vec![0.5, -0.3, 0.8];
        let original_output = original.calc(&input);
        let unoptimized_output = unoptimized.calc(&input);

        for (o, u) in std::iter::zip(original_output, unoptimized_output) {
            assert!((u - o).abs() < 1e-4);
        }
    }
}
