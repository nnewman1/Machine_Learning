# Python tutorial using Tensorflow for linear regression on a randomly generated dataset.
# Linear Regression is a common type of model for predictive analysis.
# The model is a linear approach to modeling the relationship between a scalar response (dependent variable) and explanatory variables (independent variable).
''' Linear Regression Model

y = X * beta + c + E

y = target
X = data
beta = coefficients
c = intercept
E = Error
'''

# import python libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set the seed for numpy and tensorflow for reproducibility 
np.random.seed(101)
tf.set_random_seed(101)

# Create the randomly genereated linear datasets
X_dataset = np.linspace(0, 50, 50)
Y_dataset = np.linspace(0, 50, 50)

# Add noise to the linear datasets 
X_dataset += np.random.uniform(-4, 4, 50)
Y_dataset += np.random.uniform(-4, 4, 50)

# Number of data points from the X dataset
X_size = len(X_dataset)

# Plot the X & Y datasets to visualize the data
plt.scatter(X_dataset, Y_dataset)
plt.xlabel("X_dataset")
plt.ylabel("Y_dataset")
plt.title("Training Data")
plt.show()

# Create tensorflow placeholders for the data
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Create random tensorflow variables for weights and biases
W = tf.Variable(np.random.randn(), name = "W")
B = tf.Variable(np.random.randn(), name = "B")

# Create and define hyperparameters for the model (learning rate & epochs)
learning_rate = 0.01
training_epochs = 1000

# Create the model's Hypothesis
hypothesis = tf.add(tf.multiply(X, W), B)

# Create the model's cost function
cost = tf.reduce_sum(tf.pow(hypothesis-Y, 2)) / (2 * X_size)

# Create the model's optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Create the global variables initializer
init = tf.global_variables_initializer()

# Starting the tensorflow session
with tf.Session() as sess:
	# Initializing the Variables
	sess.run(init)
	# Iterating through all the epochs
	for epoch in range(training_epochs):
		# Feeding each data point into the optimizer using a feed dictionary
		for (_X_dataset, _Y_dataset) in zip(X_dataset, Y_dataset):
			sess.run(optimizer, feed_dict = {X : _X_dataset, Y : _Y_dataset})
		# Displaying the result after every 50 epochs
		if (epoch + 1) % 50 == 0:
			# Calculating the cost at every epoch
			c = sess.run(cost, feed_dict = {X : X_dataset, Y : Y_dataset})
			print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "B =", sess.run(B))
	# Storing the necessary values to be used outside the tensorflow session
	training_cost = sess.run(cost, feed_dict = {X : X_dataset, Y : Y_dataset})
	weight = sess.run(W)
	bias = sess.run(B)

# Calculating the model's predictions
Predictions = weight * X_dataset + bias
print("\n Training Cost =", training_cost, "Weight=", weight, "Bias=", bias, '\n')

# Plot the datasets with the predictions
plt.plot(X_dataset, Y_dataset, 'ro', label ="original Data")
plt.plot(X_dataset, Predictions, label ="Fitted Line")
plt.xlabel("X_dataset")
plt.ylabel("Y_dataset")
plt.title("Linear Regression Results")
plt.legend()
plt.show()






