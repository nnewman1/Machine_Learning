# Python tutorial using pytorch for a basic Artificial Neural Network (ANN) on a sudo generated dataset.

# Import python libraries
import torch
import torch.nn as nn

# Define the input size, hidden layer size, output size and batch size
n_in, n_h, n_out, batch_size = 10, 5, 1, 10

# Define the input and target datasets (X & Y)
x = torch.randn(batch_size, n_in)
y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

# Create the ANN model
model = nn.Sequential(nn.Linear(n_in, n_h), nn.ReLU(), nn.Linear(n_h, n_out), nn.Sigmoid())

# Define the Loss Function (criterion)
criterion = torch.nn.MSELoss()
# Define the Stochastic gradient Descent (Optimizer)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# Run the model for the number of epochs
for epoch in range(50):
	# Forward pass: Compute predicted y by passing x to the model
	y_pred = model(x)
	# Compute the loss values
	loss = criterion(y_pred, y)
	# Analyze the loss values
	print('The Epoch:', epoch,' The Loss:', loss.item())
	# Zero gradients, perform backward pass, and update the weight values
	optimizer.zero_grad()
	# preform a backwards pass (Back Propagation)
	loss.backward()
	# Update the parameters
	optimizer.step()

# If the loss values are decreasing per epoch then the ANN model is learning!

