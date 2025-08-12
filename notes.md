first layer has X neurons
last layer has y neurons

each neuron in first input layer has a certain activation value between 0 and 1
each neuron in last output layer has a similar value between 0 and 1
the neural network is a function which receives an input of X parameters each with a variable value and outputs y, also having variable values

there are hidden layers in the middle
each pair of adjacent layers starting with input layer and first hidden layer, through all adjacent hidden layers and finally the last hidden layer and the output layer
have each of their neurons connected to each of the neurons in the succeeding layer
these connections are called weights, and can be negative, positive or zero
initially, all weights are set randomly

we need to calculate the value of each neuron in the first layer given all the input values, weights and also a bias for each neuron

To compute a neuron's value in the first hidden layer, sum the products of each weight connecting an input layer neuron to that hidden neuron and the corresponding input neuron's activation value

each neuron also has a bias
add this bias to the sum
initially all biases are set randomly

since the value of a weight can be negatively or positively variable, a function called the activation function is used to normalize the final value

this is done for each neuron in each layer until we reach the output layer
each neuron has a unique weight between it and each of the neurons in its preceding layer
each neuron also has a unique bias

to start with, we only have the input values
each neuron in all the other values has value zero, weights of all connections are zero and so are the biases for each neuron

we start with a training data set
one data set has a set of input values and a correct set of output values

once this single data set is fed to the neural network,
each neuron in the succeeding layers will caculate their value,
until the final layer's neuron's values are calculated

then the final layer's neuron's values are compared with the correct output values present in the data set
this is where we calculate the cost which is equal to the sum of the squares of the subtraction between each of the final layer's neuron's value and the expected correct value
the cost will be closer to zero when the final layer is close to the actual expected values and higher when it's not

the cost is calcualted as the average cost over multiple training data sets, why cant it be calculated and used to adjust weights and biases by back propgation after every training data set?

the magnitude of each of the weights and biases of each neuron in the hidden layer tells us how sensitive the cost is to each weight and bias

it will also update its biases by ?

