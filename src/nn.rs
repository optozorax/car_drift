use crate::common::pairs;
use crate::math::{fxx, Ex};
use rand::prelude::SliceRandom;
use rand::Rng;
use serde::Deserialize;
use serde::Serialize;

pub fn mul_matrix(
    input: &[fxx],
    matrix: &[fxx],
    output: &mut [fxx],
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

pub fn sigmoid(x: fxx) -> fxx {
    1. / (1. + Ex.powf(-x))
}

pub fn sqrt_sigmoid(x: fxx) -> fxx {
    x / (1. + x * x).sqrt()
}

pub fn relu10(x: fxx) -> fxx {
    x.clamp(0., 10.)
}

pub fn relu_leaky_10(x: fxx) -> fxx {
    if x > 0. { x } else { x * 0.1 }.clamp(-1., 10.)
}

pub fn relu(x: fxx) -> fxx {
    x.max(0.)
}

pub fn relu_leaky(x: fxx) -> fxx {
    if x > 0. {
        x
    } else {
        x * 0.1
    }
}

pub fn relu_leaky_smooth_10(x: fxx) -> fxx {
    if x > 10. {
        11. - 1. / (1. + x - 10.)
    } else if x < -10. {
        -1.1 + 1. / (1. - x + 10.)
    } else if x < 0. {
        x * 0.1
    } else {
        x
    }
}

fn softmax(output: &mut [fxx]) {
    let sum = output.iter().map(|x| x.exp()).sum::<fxx>();
    for x in output {
        *x = x.exp() / sum;
    }
}

fn argmax_one_hot(output: &mut [fxx]) {
    let max_index = output
        .iter()
        .enumerate()
        .max_by(|(_, &x), (_, &y)| x.partial_cmp(&y).unwrap())
        .unwrap()
        .0;
    for (i, x) in output.iter_mut().enumerate() {
        *x = if i == max_index { 1. } else { 0. };
    }
}

fn strip_bad_values(output: &mut [fxx]) {
    for x in output {
        *x = x.clamp(-1e6, 1e6);
        if x.is_nan() {
            *x = 0.;
        }
    }
}

fn sum_vectors(input: &[fxx], vector: &[fxx], output: &mut [fxx]) {
    for (out, (in1, in2)) in output.iter_mut().zip(input.iter().zip(vector.iter())) {
        *out = in1 + in2;
    }
}

fn mul_vectors(output: &mut [fxx], input: &[fxx]) {
    for (out, inp) in output.iter_mut().zip(input.iter()) {
        *out *= *inp;
    }
}

fn init_one(input: &mut [fxx]) {
    input.iter_mut().for_each(|x| *x = 1.);
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug, Copy, PartialEq, Default)]
pub enum ActivationFunction {
    #[default]
    None,
    Relu,
    Relu10,
    ReluLeaky,
    ReluLeaky10,
    ReluLeakySmooth10,
    Sigmoid,
    SqrtSigmoid,
    Softmax,
    ArgmaxOneHot,
}

impl ActivationFunction {
    fn apply_single(&self, x: fxx) -> fxx {
        match *self {
            ActivationFunction::None => x,
            ActivationFunction::Relu => relu(x),
            ActivationFunction::Relu10 => relu10(x),
            ActivationFunction::ReluLeaky => relu_leaky(x),
            ActivationFunction::ReluLeaky10 => relu_leaky_10(x),
            ActivationFunction::ReluLeakySmooth10 => relu_leaky_smooth_10(x),
            ActivationFunction::Sigmoid => sigmoid(x),
            ActivationFunction::SqrtSigmoid => sqrt_sigmoid(x),
            ActivationFunction::Softmax => panic!(),
            ActivationFunction::ArgmaxOneHot => panic!(),
        }
    }

    fn apply_vector(&self, output: &mut [fxx]) {
        match self {
            ActivationFunction::None => {}
            ActivationFunction::Relu => {
                for x in output {
                    *x = relu(*x);
                }
            }
            ActivationFunction::Relu10 => {
                for x in output {
                    *x = relu10(*x);
                }
            }
            ActivationFunction::ReluLeaky => {
                for x in output {
                    *x = relu_leaky(*x);
                }
            }
            ActivationFunction::ReluLeaky10 => {
                for x in output {
                    *x = relu_leaky_10(*x);
                }
            }
            ActivationFunction::ReluLeakySmooth10 => {
                for x in output {
                    *x = relu_leaky_smooth_10(*x);
                }
            }
            ActivationFunction::Sigmoid => {
                for x in output {
                    *x = sigmoid(*x);
                }
            }
            ActivationFunction::SqrtSigmoid => {
                for x in output {
                    *x = sqrt_sigmoid(*x);
                }
            }
            ActivationFunction::Softmax => softmax(output),
            ActivationFunction::ArgmaxOneHot => argmax_one_hot(output),
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug, PartialEq, Copy, Default)]
pub struct LayerDescription {
    pub size: usize,
    pub height: usize,
    pub func: ActivationFunction,
}

impl LayerDescription {
    pub fn new(size: usize, func: ActivationFunction) -> Self {
        Self::new_height(size, 1, func)
    }

    pub fn new_height(size: usize, height: usize, func: ActivationFunction) -> Self {
        Self { size, height, func }
    }

    pub fn none(size: usize) -> Self {
        Self::new(size, ActivationFunction::None)
    }

    pub fn relu_best(size: usize) -> Self {
        Self::new(size, ActivationFunction::ReluLeakySmooth10)
    }

    pub fn relu(size: usize) -> Self {
        Self::new(size, ActivationFunction::Relu)
    }
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct NeuralNetwork {
    sizes: Vec<LayerDescription>,
    values: Vec<fxx>,

    reserved1: Vec<fxx>,
    reserved2: Vec<fxx>,
    reserved3: Vec<fxx>,

    regularization: fxx,
    calc_count: usize,
}

impl NeuralNetwork {
    pub fn new_zeros(sizes: Vec<LayerDescription>) -> Self {
        let values_size = Self::calc_nn_len(&sizes);
        let mut result = Self {
            sizes,
            values: vec![0.; values_size],
            reserved1: Default::default(),
            reserved2: Default::default(),
            reserved3: Default::default(),
            regularization: 0.,
            calc_count: 0,
        };
        result.resize_reserved();
        result
    }

    pub fn new_params(sizes: Vec<LayerDescription>, params: &[fxx]) -> Self {
        let values_size = Self::calc_nn_len(&sizes);
        assert_eq!(params.len(), values_size);
        let mut result = Self {
            sizes,
            values: params.iter().copied().collect(),
            reserved1: Default::default(),
            reserved2: Default::default(),
            reserved3: Default::default(),
            regularization: 0.,
            calc_count: 0,
        };
        result.resize_reserved();
        result
    }

    pub fn calc_nn_len(sizes: &[LayerDescription]) -> usize {
        pairs(sizes.iter().map(|x| (x.size, x.height)))
            .map(|((a, _), (b, bh))| (a + 1) * b * bh)
            .sum()
    }

    pub fn get_sizes(&self) -> &[LayerDescription] {
        &self.sizes
    }

    pub fn get_values(&self) -> &[fxx] {
        &self.values
    }

    pub fn get_values_mut(&mut self) -> &mut [fxx] {
        &mut self.values
    }

    pub fn generate_random(sizes: Vec<LayerDescription>, rng: &mut impl Rng) -> Self {
        let mut result = Self::new_zeros(sizes);
        result
            .values
            .iter_mut()
            .for_each(|x| *x = rng.gen_range(-1.0..1.0));
        result
    }

    pub fn resize_reserved(&mut self) {
        let reserved_size = pairs(self.sizes.iter().map(|x| x.size))
            .map(|(a, b)| (a + b).max(a + a).max(b + b))
            .max()
            .unwrap_or(0);
        self.reserved1 = vec![0.; reserved_size];
        self.reserved2 = vec![0.; reserved_size];
        self.reserved3 = vec![0.; reserved_size];
    }

    pub fn calc<'a>(&'a mut self, input: &[fxx]) -> &'a [fxx] {
        let mut regularization = 0.;
        let mut values_count = 0;
        input
            .iter()
            .enumerate()
            .for_each(|(i, x)| self.reserved1[i] = *x);

        let mut offset = 0;
        for (prev, now) in pairs(self.sizes.iter()) {
            let (prev_slice, now_slice) =
                self.reserved1[..prev.size + now.size].split_at_mut(prev.size);

            init_one(&mut self.reserved3[..now.size]);
            for _ in 0..now.height {
                mul_matrix(
                    prev_slice,
                    &self.values[offset..(offset + prev.size * now.size)],
                    now_slice,
                    prev.size,
                    now.size,
                );
                offset += prev.size * now.size;
                sum_vectors(
                    now_slice,
                    &self.values[offset..(offset + now.size)],
                    &mut self.reserved2[..now.size],
                );
                offset += now.size;
                mul_vectors(&mut self.reserved3[..now.size], &self.reserved2[..now.size]);
            }

            for value in &self.reserved3[..now.size] {
                if value.abs() > 10. {
                    regularization += value.abs();
                    values_count += 1;
                }
            }
            now.func.apply_vector(&mut self.reserved3[..now.size]);
            strip_bad_values(&mut self.reserved3[..now.size]);
            std::mem::swap(&mut self.reserved1, &mut self.reserved3);
        }

        self.regularization += regularization / values_count as fxx;
        self.calc_count += 1;

        &self.reserved1[..self.sizes.last().unwrap().size]
    }

    pub fn mutate_float_value(&mut self, rng: &mut impl Rng) {
        *self.values.choose_mut(rng).unwrap() += rng.gen_range(-0.1..0.1);
    }

    pub fn get_regularization(&self) -> fxx {
        if self.calc_count == 0 {
            0.
        } else {
            self.regularization / self.calc_count as fxx
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Layer {
    pub input_size: usize,
    pub output_size: usize,
    pub stack: Vec<(Vec<Vec<fxx>>, Vec<fxx>)>, // matrix (outer vec has size output_size, inner vec has size input_size) + bias, outer array is "height", or pi-sigma network, output from each sublayer will be multiplied
    pub func: ActivationFunction,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NeuralNetworkUnoptimized {
    pub layers: Vec<Layer>,
}

impl NeuralNetworkUnoptimized {
    pub fn calc(&self, input: &[fxx]) -> Vec<fxx> {
        self.layers.iter().fold(input.to_vec(), |input, layer| {
            let mut output = vec![0.0; layer.output_size];
            for (i, output_neuron) in output.iter_mut().enumerate() {
                *output_neuron = 1.;
                for (matrix, bias) in &layer.stack {
                    *output_neuron *= matrix[i]
                        .iter()
                        .zip(input.iter())
                        .map(|(&weight, &input)| weight * input)
                        .sum::<fxx>()
                        + bias[i];
                }
                *output_neuron = layer.func.apply_single(*output_neuron);
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
                let mut stack = vec![];
                let matrix_size = input_size.size * output_size.size;
                for _ in 0..output_size.height {
                    let matrix = self.values[offset..offset + matrix_size]
                        .chunks(input_size.size)
                        .map(|row| row.to_vec())
                        .collect();
                    offset += matrix_size;
                    let bias = self.values[offset..offset + output_size.size].to_vec();
                    offset += output_size.size;
                    stack.push((matrix, bias));
                }
                Layer {
                    input_size: input_size.size,
                    output_size: output_size.size,
                    stack,
                    func: output_size.func,
                }
            })
            .collect();

        NeuralNetworkUnoptimized { layers }
    }

    pub fn from_unoptimized(unoptimized: &NeuralNetworkUnoptimized) -> Self {
        let sizes: Vec<LayerDescription> =
            std::iter::once(LayerDescription::none(unoptimized.layers[0].input_size))
                .chain(unoptimized.layers.iter().map(|layer| {
                    LayerDescription::new_height(layer.output_size, layer.stack.len(), layer.func)
                }))
                .collect();

        let values: Vec<fxx> = unoptimized
            .layers
            .iter()
            .flat_map(|layer| {
                layer
                    .stack
                    .iter()
                    .flat_map(|(matrix, bias)| matrix.iter().flatten().chain(bias.iter()))
                    .cloned()
            })
            .collect();

        let mut nn = Self {
            sizes,
            values,
            reserved1: Default::default(),
            reserved2: Default::default(),
            reserved3: Default::default(),
            regularization: 0.,
            calc_count: 0,
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
        let (matrix, bias) = layer.stack.choose_mut(rng).unwrap();
        let row = rng.gen_range(0..matrix.len());
        let col = rng.gen_range(0..matrix[row].len());
        matrix[row][col] += rng.gen_range(-0.1..0.1);
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
        for (matrix, bias) in &mut layer.stack {
            matrix.push(vec![0.0; layer.input_size]);
            bias.push(0.0);
        }

        if layer_index < self.layers.len() - 1 {
            let next_layer = &mut self.layers[layer_index + 1];
            next_layer.input_size += 1;
            for (matrix, bias) in &mut next_layer.stack {
                matrix.push(vec![0.0; next_layer.input_size]);
                bias.push(0.0);
            }
        }
    }

    pub fn add_input_neuron(&mut self) {
        let layer = &mut self.layers[0];
        layer.input_size += 1;
        for (matrix, bias) in &mut layer.stack {
            matrix.push(vec![0.0; layer.input_size]);
            bias.push(0.0);
        }
    }

    pub fn add_random_hidden_layer(&mut self, func: ActivationFunction, rng: &mut impl Rng) {
        let layer_index = rng.gen_range(0..=self.layers.len());
        self.add_hidden_layer(layer_index, func);
    }

    // layer_index = 0 - layer right after input
    // layer_index = layers.len() - layer right before output
    pub fn add_hidden_layer(&mut self, layer_index: usize, func: ActivationFunction) {
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

        let mut stack = vec![];
        let mut matrix = vec![vec![0.0; input_size]; output_size];
        #[allow(clippy::needless_range_loop)]
        for i in 0..input_size.min(output_size) {
            matrix[i][i] = 1.0;
        }
        stack.push((matrix, vec![0.0; output_size]));

        let new_layer = Layer {
            input_size,
            output_size,
            stack,
            func,
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
        for (matrix, bias) in &mut layer.stack {
            matrix.remove(neuron_index);
            bias.remove(neuron_index);
        }

        let next_layer = &mut self.layers[layer_index + 1];
        next_layer.input_size -= 1;
        for (matrix, bias) in &mut next_layer.stack {
            for row in matrix {
                row.remove(neuron_index);
            }
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
            for (matrix, bias) in &mut prev_layer.stack {
                matrix.resize(removed_layer.output_size, vec![0.0; prev_layer.input_size]);
            }
        } else {
            // Removing a hidden layer
            let removed_layer = self.layers.remove(layer_index);
            let next_layer = &mut self.layers[layer_index];

            // Adjust the input size of the next layer
            next_layer.input_size = removed_layer.input_size;

            // Resize the weight matrix of the next layer
            for (matrix, bias) in &mut next_layer.stack {
                for row in matrix {
                    row.resize(next_layer.input_size, 0.0);
                }
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
            let stack_size = rng.gen_range(1..4);
            let mut stack = vec![];
            for _ in 0..stack_size {
                let matrix = (0..output_size)
                    .map(|_| (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
                    .collect();
                let bias = (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
                stack.push((matrix, bias));
            }
            layers.push(Layer {
                input_size,
                output_size,
                stack,
                func: if i == 0 || i == neuron_counts.len() - 2 {
                    ActivationFunction::None
                } else {
                    ActivationFunction::ReluLeakySmooth10
                },
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

    fn generate_random_input(rng: &mut impl Rng, size: usize) -> Vec<fxx> {
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
                let output_before = nn.calc(&input);
                nn.add_random_hidden_layer(ActivationFunction::None, &mut rng);
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
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_neural_network_3_2_3() {
        let mut nn = NeuralNetwork::new_zeros(vec![
            LayerDescription::none(3),
            LayerDescription::relu(2),
            LayerDescription::none(3),
        ]);

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
        let mut nn = NeuralNetwork::new_zeros(vec![
            LayerDescription::none(2),
            LayerDescription::relu(2),
            LayerDescription::none(1),
        ]);

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
        let sizes = vec![
            LayerDescription::none(3),
            LayerDescription::relu(4),
            LayerDescription::none(2),
        ];
        let mut nn = NeuralNetwork::new_zeros(sizes);
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
    fn test_conversion_roundtrip_many() {
        let mut rng = StdRng::seed_from_u64(42);
        for size in 2..10 {
            for _ in 0..10 {
                let neuron_counts =
                    NeuralNetworkUnoptimized::generate_neuron_counts(&mut rng, size);
                let nn = NeuralNetworkUnoptimized::generate_random(&mut rng, &neuron_counts);
                let optimized = nn.to_optimized();
                let roundtrip = NeuralNetworkUnoptimized::from_optimized(&optimized);
                let optimized2 = roundtrip.to_optimized();

                assert_eq!(optimized.sizes, optimized2.sizes);
                assert_eq!(optimized.values, optimized2.values);
            }
        }
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
