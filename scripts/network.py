import math
import random

# activation functions, rectified linear unit or relu
# relu basically outputs the input unchanged if it is positive,
# otherwise, it outputs zero

def relu(x):
    return max(0, x)

# how do derivatives work again?
# in what context do we need the derivative of relu?

def relu_derivative(x):
    return 1 if x > 0 else 0

# softmax 
def softmax(x):
    max_x = max(x)
    exps = [math.exp(xi - max_x) for xi in x]  # Subtract max for numerical stability
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

class Addison:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.initialize_weights()

    # in a neural network all weights and biases are set randomly to start off,
    # however there are special weight initialization techniques that optimize learning,
    # Kaiming He initialization, is one that works well with relu
    # how does it work exactly?
    
    def initialize_weights(self):
        for i in range(len(self.layer_sizes) - 1):            
            
            w = [[random.gauss(0, math.sqrt(2 / self.layer_sizes[i])) 
                  for _ in range(self.layer_sizes[i + 1])] 
                 for _ in range(self.layer_sizes[i])]
            
            b = [0.0 for _ in range(self.layer_sizes[i + 1])]

            self.weights.append(w)
            self.biases.append(b)

    # forward propagation
    # ?
    # apply relu for hidden layers and softmax for output layer

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

            if i < len(self.weights) - 1:
                activations.append([relu(zj) for zj in z])
            else:
                activations.append(softmax(z))
        return activations, z_values

    # backward propagation
    # m is the number of layers in the network
    # dz is the gradient of the loss at the output layer,
    # it is calculated by ... ?
    # the main for loop starts processing each layer backwards,
    # ... ?
    # dw is the gradient of the weights for the current layer being iterated over
    # the weight gradient is then caculated by ... ?
    # the weights and biases are then updated by ... ?
    # why do we calculate gradient for next iteration for non input layers?

    def backward(self, x, y, activations, z_values, learning_rate=0.01):
        m = len(self.weights)
        
        dz = [activations[-1][i] - (1 if i == y else 0) for i in range(self.layer_sizes[-1])]
        
        for layer in range(m - 1, -1, -1):
            current_size = self.layer_sizes[layer]
            next_size = self.layer_sizes[layer + 1]
            
            dw = [[0.0 for _ in range(next_size)] for _ in range(current_size)]
            
            for i in range(current_size):
                for j in range(next_size):
                    dw[i][j] = dz[j] * activations[layer][i]
            
            for i in range(current_size):
                for j in range(next_size):
                    self.weights[layer][i][j] -= learning_rate * dw[i][j]

            for j in range(next_size):
                self.biases[layer][j] -= learning_rate * dz[j]
            
            if layer > 0:
                new_dz = [0.0] * current_size
                for i in range(current_size):
                    for j in range(next_size):

                        # for hidden layers
                        if layer < m - 1:
                            new_dz[i] += dz[j] * self.weights[layer][i][j] * relu_derivative(z_values[layer][j])
                        
                        # for output layer
                        else:
                            new_dz[i] += dz[j] * self.weights[layer][i][j]

                dz = new_dz