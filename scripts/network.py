import math
import random

# Activation Functions
def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

def softmax(x):
    max_x = max(x)
    exps = [math.exp(xi - max_x) for xi in x]  # Subtract max for numerical stability
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

# Neural Network Class
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes  # [input, hidden1, hidden2, ..., output]
        self.weights = []
        self.biases = []
        self.initialize_weights()

    def initialize_weights(self):
        for i in range(len(self.layer_sizes) - 1):
            # He initialization for ReLU
            w = [[random.gauss(0, math.sqrt(2 / self.layer_sizes[i])) 
                  for _ in range(self.layer_sizes[i + 1])] 
                 for _ in range(self.layer_sizes[i])]
            b = [0.0 for _ in range(self.layer_sizes[i + 1])]
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        activations = [x]
        z_values = []
        for i in range(len(self.weights)):
            z = [0.0] * self.layer_sizes[i + 1]
            for j in range(self.layer_sizes[i + 1]):
                for k in range(self.layer_sizes[i]):
                    z[j] += self.weights[i][k][j] * activations[-1][k]
                z[j] += self.biases[i][j]
            z_values.append(z)
            # Apply ReLU for hidden layers, softmax for output
            if i < len(self.weights) - 1:
                activations.append([relu(zj) for zj in z])
            else:
                activations.append(softmax(z))
        return activations, z_values

    def backward(self, x, y, activations, z_values, learning_rate=0.01):
        m = len(self.weights)  # number of layers
        
        # Initialize gradients for output layer
        dz = [activations[-1][i] - (1 if i == y else 0) for i in range(self.layer_sizes[-1])]
        
        # Process each layer backwards
        for layer in range(m - 1, -1, -1):
            # Current layer sizes
            current_size = self.layer_sizes[layer]
            next_size = self.layer_sizes[layer + 1]
            
            # Initialize weight gradients for current layer
            dw = [[0.0 for _ in range(next_size)] for _ in range(current_size)]
            
            # Calculate weight gradients
            for i in range(current_size):
                for j in range(next_size):
                    dw[i][j] = dz[j] * activations[layer][i]
            
            # Update weights and biases
            for i in range(current_size):
                for j in range(next_size):
                    self.weights[layer][i][j] -= learning_rate * dw[i][j]
            for j in range(next_size):
                self.biases[layer][j] -= learning_rate * dz[j]
            
            # Calculate gradients for next iteration (if not at input layer)
            if layer > 0:
                new_dz = [0.0] * current_size
                for i in range(current_size):
                    for j in range(next_size):
                        if layer < m - 1:  # For hidden layers
                            new_dz[i] += dz[j] * self.weights[layer][i][j] * relu_derivative(z_values[layer][j])
                        else:  # For output layer
                            new_dz[i] += dz[j] * self.weights[layer][i][j]
                dz = new_dz
