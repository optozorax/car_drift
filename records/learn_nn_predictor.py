import json
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import tensorflow as tf
import glob

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


def load_and_prepare_data_world_model():
    all_X = []  # Will contain [state, action] pairs
    all_Y = []  # Will contain next states
    
    # Get all matching JSON files
    json_files = glob.glob("record*.json")
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            trajectory = json.load(f)
            
            # Each trajectory is a list of (state, action) pairs
            for i in range(len(trajectory) - 1):  # -1 because we need next state
                current_state, current_action = trajectory[i]
                next_state, _ = trajectory[i + 1]  # We don't need the next action
                
                # Combine current state and action for input
                X_entry = np.concatenate([current_state, current_action])
                Y_entry = next_state
                
                all_X.append(X_entry)
                all_Y.append(Y_entry)
    
    # Convert to numpy arrays
    X = np.array(all_X)
    Y = np.array(all_Y)
    
    return X, Y

X_train, y_train = load_and_prepare_data_world_model()

batch_size = len(X_train)

# Create the model
model = keras.Sequential([
    layers.Input(shape=(58,)),
    layers.Dense(60, activation='relu'),
    layers.Dense(52, activation='relu')
])

# Compile the model
# model.compile(optimizer='adam',loss=custom_loss)
model.compile(optimizer='adam',loss='mse')

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
    convert_weights_to_json(model, 'neural_network_predictor.json')
    print("Model saved!")
