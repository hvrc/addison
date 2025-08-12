import sys
import pickle
from network import NeuralNetwork

def create_input_vector(x1, x2):
    x = [0.0] * 20
    idx1 = int(x1)
    idx2 = int(x2) + 10
    x[idx1] = 1.0
    x[idx2] = 1.0
    return x

def predict(x1, x2):
    with open('../model/model.pkl', 'rb') as f:
        nn = pickle.load(f)
    x = create_input_vector(x1, x2)
    activations, _ = nn.forward(x)
    prediction = activations[-1].index(max(activations[-1]))
    return prediction

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python predict.py <x1> <x2>")
        sys.exit(1)
    x1, x2 = float(sys.argv[1]), float(sys.argv[2])
    print(predict(x1, x2))