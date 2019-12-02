# Short Python tutorial using numpy with a one hidden layer Artificial Neural Network (ANN) on a sudo generated dataset.
# An Artificial Neural Network is based on the structure of a biological brain. 
# These systems learn to perform tasks or classify based on data, without the need to be programmed specific task rules.
# Python is an interpreted, high-level, general-purpose programming language.
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

# Import python libraries
from numpy import exp, array, random, dot
# Define training set inputs
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
# Define training set outputs
training_set_outputs = array([[0, 1, 1, 0]]).T
# Define the seed for reproducibility
random.seed(1)
# Define a single neuron, with 3 input connections and 1 output connection
synaptic_weights = 2 * random.random((3, 1)) - 1
# Anaylze the random starting synaptic weights
print ("Random starting synaptic weights: \n", synaptic_weights, '\n')
# Run the ANN model 10000 times
for iteration in range(10000):
	# Pass the training set through our neural network (a single neuron).
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    # We train the neural network's weights through a process of trial and error.
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
# Anaylze the random synaptic weights after training
print ("New synaptic weights after training: \n", synaptic_weights, '\n')
# Test and anaylze the neural network with a new situation.
print ("Considering new situation [1, 0, 0] -> Accuracy:", 1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))))
