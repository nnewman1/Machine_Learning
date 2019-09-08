# Python tutorial using numpy for forward propagation used by ANN algorithms on a sudo dataset.
# The input data is fed in the forward direction through the network. 
# Each hidden layer accepts the input data, processes it as per the activation function and passes to the successive layer.

# Import the Numpy library
import numpy as np

# Create the X dataset
x = np.array([1.0, 0.7, 0.3]).reshape((3,1))
print('x: ==> ','\n',x, '\n')
# Create the Y dataset
y = np.array([1.0]).reshape((1,1))
print('y: ==> ','\n',y, '\n')
# Create size variables for the X & Y Dataset
N_x = np.size(x)
N_y = np.size(y)
# Create the number of nodes each hidden layer will have
N_h0 = 3
N_h1 = 2 

# Initalize the weights for each hidden and output layer
W_h0 = np.random.random((N_h0, N_x))
print('W_h0: ==> ','\n',W_h0, '\n')
N_h0r = np.size(W_h0,0)
W_h1 = np.random.random((N_h1, N_h0r))
print('W_h1: ==> ','\n',W_h1, '\n')
N_h1r = np.size(W_h1,0)
W_y = np.random.random((N_y, N_h1r))
print('W_y: ==> ','\n',W_y, '\n')

# Define the Sigmoid activation functiuon
def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

# Step one, calculate the sum of Hidden layer 0
Z_h0 = np.matmul(W_h0, x)
print('Z_h0: ==> ','\n',Z_h0, '\n')
# Step two, calculate activation for hidden layer 0 
h0 = sigmoid(Z_h0)
print('H0: ==> ','\n',h0, '\n')
# Step three, calculate the sum of hidden layer 1
Z_h1 = np.matmul(W_h1, h0)
print('Z_h1: ==> ','\n',Z_h1, '\n')
# Step four, calculate activation for hidden layer 1
h1 = sigmoid(Z_h1)
print('H1: ==> ','\n',h1, '\n')
# Step five, calculate the sum for the output layer
Z_y = np.matmul(W_y, h1)
print('Z_y: ==> ','\n',Z_y, '\n')
# Step six, calculate activation for the output layer
y_hat = sigmoid(Z_y)
print('y_hat: ==> ','\n',y_hat, '\n')
# Step seven, calculate error between Y_hat and predicted labels Y
error_y = (y - y_hat)
print('error_y: ==> ','\n',error_y)





