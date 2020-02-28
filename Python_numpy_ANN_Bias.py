# Python tutorial using numpy with a one hidden layer Artificial Neural Network (ANN) with bias on a sudo generated dataset.
# An Artificial Neural Network is based on the structure of a biological brain. 
# These systems learn to perform tasks or classify based on data, without the need to be programmed specific task rules.
# Python is an interpreted, high-level, general-purpose programming language.
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
# The Python Math Library provides access to common math functions and constants in Python.

# Import python libraries
import math
import random

# Define the neuralNetwork Class
class NeuralNetwork():
	def __init__(self):

		# Seed the random number generator, so we get the same random numbers each time
		random.seed(1)

		# Create 3 weights and set them to random values in the range -1 to +1
		self.weights = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]

		# Create a bias and set it to a random value in the range -1 to +1
		self.bias = random.uniform(-1, 1)

	# Make a prediction
	def think(self, neuron_inputs):
		sum_of_weighted_inputs = self.__sum_of_weighted_inputs(neuron_inputs)
		neuron_output = self.__sigmoid(sum_of_weighted_inputs + self.bias) 
		return neuron_output

	# Adjust the weights of the neural network to minimise the error for the training set
	def train(self, training_set_examples, number_of_iterations):
		for iteration in range(number_of_iterations):
			for training_set_example in training_set_examples:

				# Predict the output based on the training set example inputs
				predicted_output = self.think(training_set_example["inputs"])

				# Calculate the error as the difference between the desired output and the predicted output
				error_in_output = training_set_example["output"] - predicted_output

				# Iterate through the weights and adjust each one
				for index in range(len(self.weights)):

					# Get the neuron's input associated with this weight
					neuron_input = training_set_example["inputs"][index]

					# Calculate how much to adjust the weights by using the delta rule (gradient descent)
					adjust_weight = neuron_input * error_in_output * self.__sigmoid_gradient(predicted_output)

					# Adjust the weight
					self.weights[index] += adjust_weight

				# Adjust the bias
				self.bias += error_in_output * self.__sigmoid_gradient(predicted_output)


	# Calculate the sigmoid (our activation function)
	def __sigmoid(self, sum_of_weighted_inputs):
		return 1 / (1 + math.exp(-sum_of_weighted_inputs))

	# Calculate the gradient of the sigmoid using its own output	
	def __sigmoid_gradient(self, neuron_output):
		return neuron_output * (1 - neuron_output)

	# Multiply each input by its own weight, and then sum the total
	def __sum_of_weighted_inputs(self, neuron_inputs):
		sum_of_weighted_inputs = 0
		for index, neuron_input in enumerate(neuron_inputs):
			sum_of_weighted_inputs += self.weights[index] * neuron_input
		return sum_of_weighted_inputs

# Intialise a single layer neural network.
neural_network = NeuralNetwork()

# Analyze the starting weights and bias
print("\nRandom starting weights: ", str(neural_network.weights))
print("Random starting bias: ", str(neural_network.bias), '\n')

# The neural network will use this training set of 4 examples, to learn the pattern
training_set_examples = [{"inputs": [0, 0, 1], "output": 0},
						{"inputs": [1, 1, 1], "output": 1},
						{"inputs": [1, 0, 1], "output": 1},
						{"inputs": [0, 1, 1], "output": 0}]

# Train the neural network using 10,000 iterations
neural_network.train(training_set_examples, number_of_iterations=10000)

# Analyze the weights and bias after training
print("\nNew weights after training: ", str(neural_network.weights))
print("New bias after training: ", str(neural_network.bias), '\n')

# Make a prediction for a situation
new_situation = [0, 0, 1]
prediction = neural_network.think(new_situation)

# Test and anaylze the neural network with a new situation.
print("Prediction for the situation ", str(new_situation), " -> ", str(prediction))

