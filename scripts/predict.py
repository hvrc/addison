import sys
import pickle
from network import Addison

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

x1, x2 = float(sys.argv[1]), float(sys.argv[2])
print(predict(x1, x2))