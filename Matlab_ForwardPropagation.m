% Matlab tutorial for forward propagation used by ANN algorithms on a sudo dataset.
% The input data is fed in the forward direction through the network. 
% Each hidden layer accepts the input data, processes it as per the activation function and passes to the successive layer.
% MATLAB is a high-performance language for technical computing, algorithm development, modeling, simulation, and prototyping for math and computation.
% An Artificial Neural Network is based on the structure of a biological brain. 
% These systems learn to perform tasks or classify based on data, without the need to be programmed specific task rules.

clc;
clear;

% Create the X dataset
x = [1 0.7 0.3]'; % input values
% Create the Y dataset
y = [1.0]'; % Output labels for prediction
% Create size variables for the X & Y Dataset
N_x = size(x,1); % X row size
N_y = size(y,1); % Y row size
% Create the number of nodes each hidden layer will have
N_h0 = 3; % Hidden Layer 0 Nodes 
N_h1 = 2; % Hidden Layer 1 Nodes

% Initalize the weights for each hidden and output layer
W_h0  = rand(N_h0,N_x); % Weights for hidden layer 0 
N_h0r = size(W_h0,1); % Hidden Layer 0 row size
W_h1 = rand(N_h1,N_h0r); % Weights for hidden layer 1
N_h1r = size(W_h1,1); % Hidden Layer 1 row size
W_y  = rand(N_y,N_h1r); % weights for output layer

% Step one, calculate the sum of Hidden layer 0
Z_h0 = W_h0 * x;
Z_h0 = Z_h0';

% Step two, calculate activation for hidden layer 0 
h0 = Sigmoid(Z_h0);
h0 = h0';

% Step three, calculate the sum of hidden layer 1
Z_h1 = W_h1 * h0;
Z_h1 = Z_h1';

% Step four, calculate activation for hidden layer 1
h1 = Sigmoid(Z_h1);
h1 = h1';

% Step five, calculate the sum for the output layer
Z_y = W_y * h1;
Z_y = Z_y';

% Step six, calculate activation for the output layer
y_hat = Sigmoid(Z_y);
y_hat = y_hat';

% Step seven, calculate error between Y_hat and predicted labels Y
error_y = (y-y_hat);






