import numpy as np

# Example input and output data
inputs = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
])

outputs = np.array([[0], [1], [1], [0]])

# Initialize weights with random values
np.random.seed(1)
weights = 2 * np.random.random((3, 1)) - 1

print("Initial weights:")
print(weights)

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)


# Training the neural network
for i in range(500):  # Number of training iterations
    input_layer = inputs
    feed_outputs = sigmoid(np.dot(input_layer, weights))
    
    error = outputs - feed_outputs
    
    adjustments = error * sigmoid_derivative(feed_outputs)
    
    weights += np.dot(input_layer.T, adjustments)


print("Weights after training:")
print(weights)

print("Output after training:")
print(feed_outputs.round(1))
