import json
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import tensorflow as tf
import glob

def custom_loss(y_true, y_pred):
    # Define the sqrt_sigmoid function
    def sqrt_sigmoid(x):
        return x / tf.sqrt(1. + x * x)
    
    # Define the two_relus_to_ratio function
    def two_relus_to_ratio(a, b):
        condition1 = tf.logical_and(a > b, a > 0)
        condition2 = tf.logical_and(b > a, b > 0)
        zeros = tf.zeros_like(a)
        result_a = sqrt_sigmoid(a)
        result_b = -sqrt_sigmoid(b)
        output = tf.where(condition1, result_a, tf.where(condition2, result_b, zeros))
        return output

    # Transform y_true
    y_true_0 = tf.maximum(sqrt_sigmoid(y_true[:, 0]), 0.)
    y_true_1 = two_relus_to_ratio(y_true[:, 1], y_true[:, 2])
    y_true_2 = tf.maximum(sqrt_sigmoid(y_true[:, 3]), 0.)
    y_true_3 = two_relus_to_ratio(y_true[:, 4], y_true[:, 5])
    y_true_transformed = tf.stack([y_true_0, y_true_1, y_true_2, y_true_3], axis=1)
    
    # Transform y_pred
    y_pred_0 = tf.maximum(sqrt_sigmoid(y_pred[:, 0]), 0.)
    y_pred_1 = two_relus_to_ratio(y_pred[:, 1], y_pred[:, 2])
    y_pred_2 = tf.maximum(sqrt_sigmoid(y_pred[:, 3]), 0.)
    y_pred_3 = two_relus_to_ratio(y_pred[:, 4], y_pred[:, 5])
    y_pred_transformed = tf.stack([y_pred_0, y_pred_1, y_pred_2, y_pred_3], axis=1)
    
    # Compute mean squared error per sample
    mse = tf.reduce_mean(tf.square(y_true_transformed - y_pred_transformed), axis=1)
    return mse


def convert_weights_to_json(model, name):
    layers = []
    
    # Get all layers with weights (Dense layers)
    for layer in model.layers:
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
    with open(name, 'w') as f:
        json.dump({"layers": layers}, f, indent=2)


class Callback(tf.keras.callbacks.Callback):
    SHOW_NUMBER = 10
    counter = 0

    def __init__(self, start):
        self.counter = start

    def on_train_batch_end(self, batch, logs=None):
        if (self.counter % self.SHOW_NUMBER) == 0:
            print('Epoch: ' + str(self.counter) + ' loss: ' + str(logs['loss']))
        self.counter += 1


def load_and_prepare_data_car():
    # Load the data from JSON file
    data = []
    for filename in glob.glob("record_*.json"):
        with open(filename, 'r') as file:
            file_data = json.load(file)
            data.extend(file_data)

    # Convert data to numpy arrays
    X = np.array([pair[0] for pair in data])  # Input features
    y = np.array([pair[1] for pair in data])  # Target values

    return X, y

X_train, y_train = load_and_prepare_data_car()

batch_size = len(X_train)

# Create the model
model = keras.Sequential([
    layers.Input(shape=(52,)),
    layers.Dense(10, activation='relu'),
    layers.Dense(6, activation='relu')
])

# Compile the model
model.compile(optimizer='adam',loss=custom_loss)

# Display model summary
model.summary()

for i in range(100):
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=100,
        callbacks=[Callback(i * 100)],
        verbose=0,
    )

    # Call this after training
    convert_weights_to_json(model, 'neural_network_actor.json')
    print("Model saved!")
