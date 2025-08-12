import random
import pickle
import math
from network import NeuralNetwork

def create_input_vector(x1, x2):
    x = [0.0] * 20
    idx1 = int(x1)
    idx2 = int(x2) + 10
    x[idx1] = 1.0
    x[idx2] = 1.0
    return x

def train_network(batch_size=100, epochs=500, learning_rate=0.01):
    with open('../model/model.pkl', 'rb') as f:
        nn = pickle.load(f)
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, 10000, batch_size):  # 10,000 samples per epoch
            batch_x = []
            batch_y = []
            for _ in range(batch_size):
                x1 = random.randint(0, 9)
                x2 = random.randint(0, 9)
                y = x1 + x2
                batch_x.append(create_input_vector(x1, x2))
                batch_y.append(y)
            
            # Process batch
            loss = 0
            for x, y in zip(batch_x, batch_y):
                activations, z_values = nn.forward(x)
                loss -= math.log(activations[-1][y] + 1e-10)  # Cross-entropy loss
                nn.backward(x, y, activations, z_values, learning_rate / batch_size)
            
            total_loss += loss / batch_size
            print(f"Epoch {epoch+1}, Iteration {i//batch_size + 1}, Loss: {total_loss / (i//batch_size + 1):.4f}")
        
        # Save model after each epoch
        with open('../model/model.pkl', 'wb') as f:
            pickle.dump(nn, f)

if __name__ == '__main__':
    train_network()