import sys
from network import NeuralNetwork
import pickle

def create_network(hidden_layers):
    input_size = 20   # 10 neurons for each number (0-9)
    output_size = 19  # Sums 0 to 18
    layer_sizes = [input_size] + hidden_layers + [output_size]
    nn = NeuralNetwork(layer_sizes)
    with open('../model/model.pkl', 'wb') as f:
        pickle.dump(nn, f)
    print(f"Created neural network with layers: {layer_sizes}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python create.py <num_hidden_layers> <neurons_layer1> <neurons_layer2> ...")
        sys.exit(1)
    hidden_layers = [int(x) for x in sys.argv[2:]]
    create_network(hidden_layers)