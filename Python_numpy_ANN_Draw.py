# Python tutorial using numpy with a one hidden layer Artificial Neural Network (ANN) with plots on a sudo generated dataset.
# An Artificial Neural Network is based on the structure of a biological brain. 
# These systems learn to perform tasks or classify based on data, without the need to be programmed specific task rules.
# Python is an interpreted, high-level, general-purpose programming language.
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

# Import python libraries
from matplotlib import pyplot
from math import cos, sin, atan

# Define the Neuron Class
class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)

# Define the Layer Class
class Layer():
    def __init__(self, network, number_of_neurons):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment))
        pyplot.gca().add_line(line)

    def draw(self):
        for neuron in self.neurons:
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron)

# Define the NeuralNetwork Class
class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons):
        layer = Layer(self, number_of_neurons)
        self.layers.append(layer)

    def draw(self):
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.show()

# Define the Main
if __name__ == "__main__":
    # Define the vertical distance between layers on the graph
    vertical_distance_between_layers = 6
    # Define the horizontal distance between laters on the graph
    horizontal_distance_between_neurons = 2
    # Define the size of the neuron radius
    neuron_radius = 0.5
    # Define the number if neurons in the biggest layer
    number_of_neurons_in_widest_layer = 4
    # Create the NeuralNetwork object
    network = NeuralNetwork()
    # Create the input layer of the graph with 3 nodes
    network.add_layer(3)
    # Create the hidden layer of the graph with 4 nodes
    network.add_layer(4)
    # Create the output layer of the graph with 1 node
    network.add_layer(1)
    # Draw the recently created ANN graph
    network.draw()

