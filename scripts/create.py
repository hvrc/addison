import sys
from network import Addison
import pickle

# the input is two numbers between 0 and 9
# so the first layer is 20 neurons,
# two of them will be hot (1.0), the rest will be cold (0.0)
# our output layer will have 19 neurons,
# for all possible sums of single digit number (0 to 18)

def create_network(hidden_layers):
    input_size = 20
    output_size = 19
    layer_sizes = [input_size] + hidden_layers + [output_size]
    nn = Addison(layer_sizes)

    with open('../model/model.pkl', 'wb') as f:
        pickle.dump(nn, f)

hidden_layers = [int(x) for x in sys.argv[2:]]
create_network(hidden_layers)