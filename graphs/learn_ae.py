import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

tf.config.threading.set_inter_op_parallelism_threads(3)

# Load data with explicit encoding
with open('tracks_dirs_converted.json', 'r', encoding='utf-8') as file:
    data = np.array(json.load(file), dtype=np.float32)

train_data = data

# Define the autoencoder architecture
input_dim = 21
encoding_dim = 10
hidden_dim = 10

# 1 -> 0.045
# 10 1 -> 0.1655
# 21 -> 0.0043
# 5 -> 0.0196
# 10 5 -> 0.0187
# 15 10 7 5 -> 0.0281
# 10 10 10 5 -> 0.0101
# 40 5 -> 0.0117



class SymmetricAutoencoder(tf.keras.Model):
    def __init__(self, encoder, decoder, symmetry_weight=1.0):
        super(SymmetricAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.symmetry_weight = symmetry_weight
        
        # Add metrics
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.symmetry_loss_tracker = keras.metrics.Mean(name="symmetry_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.symmetry_loss_tracker,
        ]
    
    def compile(self, optimizer, loss='mse', **kwargs):
        super(SymmetricAutoencoder, self).compile(
            optimizer=optimizer,
            loss=loss,
            **kwargs
        )
        
    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
    
    @tf.function
    def train_step(self, data):
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data
        
        with tf.GradientTape() as tape:
            # Regular forward pass
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            
            # Symmetric constraints
            reversed_input = tf.reverse(x, axis=[1])
            encoded_reversed = self.encoder(reversed_input)
            decoded_reversed = self.decoder(encoded_reversed)
            
            # Calculate losses
            reconstruction_loss = tf.reduce_mean(tf.square(x - decoded))
            
            encoder_symmetry_loss = tf.reduce_mean(
                tf.square(tf.reverse(encoded, axis=[1]) - encoded_reversed)
            )
            
            decoder_symmetry_loss = tf.reduce_mean(
                tf.square(tf.reverse(decoded, axis=[1]) - decoded_reversed)
            )
            
            symmetry_loss = encoder_symmetry_loss + decoder_symmetry_loss
            total_loss = reconstruction_loss + self.symmetry_weight * symmetry_loss
        
        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.symmetry_loss_tracker.update_state(symmetry_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "symmetry_loss": self.symmetry_loss_tracker.result()
        }


class KeepMaxActivation(keras.layers.Layer):
    def __init__(self, max_penalty=0.1, warmup_epochs=10, **kwargs):
        super().__init__(**kwargs)
        self.max_penalty = max_penalty
        self.warmup_epochs = warmup_epochs
        self.current_epoch = tf.Variable(0.0, trainable=False)
    
    def build(self, input_shape):
        self.activation_count = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=False,
            name='activation_count'
        )
        super().build(input_shape)
    
    def call(self, inputs):
        # Calculate current penalty strength
        penalty_factor = tf.minimum(self.current_epoch / self.warmup_epochs, 1.0)
        current_penalty = self.max_penalty * penalty_factor

        max_values = tf.reduce_max(inputs, axis=1, keepdims=True)
        mask = tf.cast(tf.equal(inputs, max_values), tf.float32)
        output = inputs * mask
        
        # Update activation counts
        batch_activations = tf.reduce_mean(mask, axis=0)
        self.activation_count.assign(0.9 * self.activation_count + 0.1 * batch_activations)
        
        # Diversity penalty
        diversity_penalty = tf.reduce_sum(tf.square(self.activation_count))
        self.add_loss(current_penalty * diversity_penalty)
        
        # # Entropy penalty
        # probs = tf.reduce_mean(mask, axis=0)
        # entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10))
        # self.add_loss(-current_penalty * entropy)
        
        # L1 regularization
        l1_loss = tf.reduce_mean(tf.abs(output))
        self.add_loss(current_penalty * l1_loss)
        
        return output

class NeuronUsageCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % 100 == 0:  # Print every 10 epochs
            activation_counts = self.model.layers[0].layers[-1].activation_count.numpy()
            print("\nNeuron usage distribution:", activation_counts)
            print("Entropy:", -np.sum(activation_counts * np.log(activation_counts + 1e-10)))


# Create the encoder
encoder = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),
    # keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(encoding_dim, activation='relu'),
    KeepMaxActivation(max_penalty=1.0, warmup_epochs=100),
])

# Create the decoder
decoder = keras.Sequential([
    keras.layers.Input(shape=(encoding_dim,)),
    # keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(input_dim)
])

optimizer = keras.optimizers.Adam()
optimizer.learning_rate.assign(0.01)

# Create the autoencoder
autoencoder = keras.Sequential([encoder, decoder])
autoencoder.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

# autoencoder = SymmetricAutoencoder(encoder, decoder, symmetry_weight=1.0)
# autoencoder.compile(optimizer='adam')

# autoencoder = keras.models.load_model('autoencoder_model.keras')

# Custom training loop
history = autoencoder.fit(
    train_data, train_data,  # Input = Output for autoencoder
    epochs=1000,  # You can adjust this
    batch_size=len(train_data),  # You can adjust this
    shuffle=True,
    callbacks=[NeuronUsageCallback()],
)

def convert_weights_to_json(autoencoder):
    layers = []
    
    # Get all layers with weights (Dense layers)
    weight_layers = [l for l in autoencoder.layers if isinstance(l, keras.Sequential)]
    for sequential in weight_layers:
        for layer in sequential.layers:
            if isinstance(layer, keras.layers.Dense):
                weights, bias = layer.get_weights()
                # Convert weights to correct format (output_size x input_size)
                weights = weights.T.tolist()  # Transpose to match Rust format
                layer_dict = {
                    "input_size": weights[0].__len__(),
                    "output_size": len(weights),
                    "matrix": weights,
                    "bias": bias.tolist()
                }
                layers.append(layer_dict)

    # Save to file
    with open('neural_network.json', 'w') as f:
        json.dump({"layers": layers}, f, indent=2)


# Call this after training
convert_weights_to_json(autoencoder)

# autoencoder.save('autoencoder_model.keras')
print("Model is saved to autoencoder_model.keras")

# Make predictions on the full dataset
predictions = autoencoder.predict(data, verbose=0)

# Print final loss
final_loss = np.mean((predictions - data) ** 2)
print(f"Final MSE Loss: {final_loss:.4f}")

# Save predictions to file
with open('predictions.json', 'w') as f:
    json.dump(predictions.tolist(), f)
