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
    for filename in glob.glob("record_smooth_left_and_right_2024_11_04__23_10_50.json"):
        with open(filename, 'r') as file:
            file_data = json.load(file)
            data.extend(file_data)

    # Convert data to numpy arrays
    X = np.array([pair[0] for pair in data])  # Input features
    y = np.array([pair[1] for pair in data])  # Target values

    return X, y

def transform_to_car_input(y):
    # Convert array of 6 numbers to array of 4 numbers (CarInput format)
    # Based on CarInput::from_f32 implementation in Rust
    
    def sqrt_sigmoid(x):
        # Helper function to match Rust implementation
        return x / np.sqrt(1. + x * x)
    
    def two_relus_to_ratio(a, b):
        # Helper function matching Rust implementation
        if a > b and a > 0:
            return sqrt_sigmoid(a)
        elif b > a and b > 0:
            return -sqrt_sigmoid(b)
        else:
            return 0
    
    # Input array should have 6 elements
    assert y.shape[1] == 6
    
    # Create output array with 4 elements per input
    result = np.zeros((y.shape[0], 4))
    
    # brake: sqrt_sigmoid(input[0]).max(0.)
    result[:, 0] = np.maximum(0, np.array([sqrt_sigmoid(x) for x in y[:, 0]]))
    
    # acceleration: two_relus_to_ratio(input[1], input[2])
    result[:, 1] = np.array([two_relus_to_ratio(a, b) for a, b in zip(y[:, 1], y[:, 2])])
    
    # remove_turn: sqrt_sigmoid(input[3]).max(0.)
    result[:, 2] = np.maximum(0, np.array([sqrt_sigmoid(x) for x in y[:, 3]]))
    
    # turn: two_relus_to_ratio(input[4], input[5])
    result[:, 3] = np.array([two_relus_to_ratio(a, b) for a, b in zip(y[:, 4], y[:, 5])])
    
    return result

# Define possible actions as array of tuples (brake, acceleration, remove_turn, turn)
POSSIBLE_ACTIONS = [
    (0., 0., 0., 0.),    # 0, No action
    (1., 0., 0., 0.),    # 1, Brake only
    (0., 1., 0., 0.),    # 2, Forward
    (0., -1., 0., 0.),   # 3, Backward
    (0., 0., 1., 0.),    # 4, Remove turn only
    (0., 0., 0., 1.),    # 5, Turn right
    (0., 0., 0., -1.),   # 6, Turn left
    (0., 1., 0., 1.),    # 7, Forward + right
    (0., 1., 0., -1.),   # 8, Forward + left
    (0., 1., 1., 0.),    # 9, Forward + remove turn
    (1., 0., 1., 0.),    # 10, Brake + remove turn

    (1., 1., 1., 0.),    # 11, Brake + forward + remove turn
    (1., 1., 0., 1.),    # 12, Brake + forward + turn right
    (1., 1., 0., -1.),   # 13, Brake + forward + turn left
]

def find_nearest_action_index(car_input):
    """Find index of nearest action from POSSIBLE_ACTIONS for a single car input"""
    min_dist = float('inf')
    best_idx = 0
    
    # Convert car input to array for easier comparison
    input_array = np.array([
        car_input[0], # brake
        car_input[1], # acceleration  
        car_input[2], # remove_turn
        car_input[3]  # turn
    ])
    
    for i, action in enumerate(POSSIBLE_ACTIONS):
        action_array = np.array(action)
        
        # Calculate Euclidean distance
        dist = np.sum((input_array - action_array) ** 2)
        if dist < min_dist:
            min_dist = dist
            best_idx = i

    if best_idx == 11:
        return 4 # Remove turn only

    if best_idx == 12:
        return 5 # Turn right

    if best_idx == 13:
        return 6 # Turn left

    # if min_dist > 0.05:
    #     print(min_dist, best_idx, car_input)
            
    return best_idx

def convert_to_action_indices(y_car_input):
    """Convert array of car inputs to array of action indices"""
    return np.array([find_nearest_action_index(y) for y in y_car_input])


X_train, y_train = load_and_prepare_data_car()

def classification_approach():
    # Convert raw outputs to car input format
    y_car_input = transform_to_car_input(y_train)

    # Convert car inputs to action indices
    y_train_categorical = convert_to_action_indices(y_car_input)

    # Convert to one-hot encoding
    y_train_result = keras.utils.to_categorical(y_train_categorical, num_classes=11)

    # Create the classification model
    model = keras.Sequential([
        layers.Input(shape=(52,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(11, activation='softmax')
    ])

    # Compile the model with categorical crossentropy loss
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return y_train_result, model

def regression_approach():
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

    return y_train,model

batch_size = len(X_train)
y_train_result, model = classification_approach()

for i in range(100):
    # Train the model
    history = model.fit(
        X_train,
        y_train_result,
        batch_size=batch_size,
        epochs=100,
        callbacks=[Callback(i * 100)],
        verbose=0,
    )

    # Call this after training
    convert_weights_to_json(model, 'neural_network_actor.json')
    print("Model saved!")
