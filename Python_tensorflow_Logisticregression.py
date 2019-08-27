# Python tutorial using Tensorflow for Logistic Regression using the iris dataset

# Import Python Libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Define the Sigmoid activation function
def sigmoid(z):
	return 1 / (1 + np.exp(-z))

# Plot the Sigmoid activation function
#plt.plot(np.arange(-5, 5, 0.1), sigmoid(np.arange(-5, 5, 0.1)))
#plt.title("Visualization of the Sigmoid function")
#plt.show()

# Import the Iris Dataset
dataSet = load_iris()

# Analyze the first five entires of the dataset's values
#print(dataSet.data[0:5])
# Analyze the feature set of the data
#print("features: ", dataSet.feature_names)
# Analyze the target set of the data
#print("Labels: ", dataSet.target_names)
# Analyze the dataset's shape
#print("Dataset Shape: ", dataSet.data.shape)
# Analyze the target set of the data
#print("Dataset Labels: ", dataSet.target)
 
'''
# Plot the Iris dataset for visualization
# Positive Data Points 
x_pos = np.array([dataSet.data[i] for i in range(len(dataSet.data)) if dataSet.target[i] == 1])
# Negative Data Points 
x_neg = np.array([dataSet.data[i] for i in range(len(dataSet.data)) if dataSet.target[i] == 0])
# Plotting the Positive Data Points
plt.scatter(x_pos[:, 0], x_pos[:, 1], color = 'blue', label = 'Positive')
# Plotting the Negative Data Points
plt.scatter(x_neg[:, 0], x_neg[:, 1], color = 'red', label = 'Negative')
plt.xlabel('Feature 1') 
plt.ylabel('Feature 2')
plt.title('Plot of given data')
plt.legend()
plt.show()
'''

# Create the One Hot Encoder for data preparation 
theEncoder = OneHotEncoder()

# Fit and Apply the Encoder to the X and Y datasets
theEncoder.fit(dataSet.data)
X_dataset = theEncoder.transform(dataSet.data).toarray()
theY = dataSet.target
theY = theY.reshape(-1,1)
theEncoder.fit(theY)
Y_dataset = theEncoder.transform(theY).toarray()

# Create and define hyperparameters for the model (learning rate & epochs)
alpha, epochs = 0.0035, 300
m, n = X_dataset.shape
#print("m =", m)
#print("n =", n)
#print("Learning Rate =", alpha)
#print("Number of Epochs =", epochs)

# Create tensorflow placeholders for the data
X = tf.placeholder(tf.float32, [None, n])
Y = tf.placeholder(tf.float32, [None, 3])
# Create random tensorflow variables for weights and biases
W = tf.Variable(tf.zeros([n, 3]))
B = tf.Variable(tf.zeros([3]))

# Create the model's Hypothesis
Hypothesis = tf.nn.sigmoid(tf.add(tf.matmul(X, W), B))
# Create the model's cost function
Cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=Hypothesis, labels=Y)
# Create the model's optimizer
Optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(Cost)
# Create the global variables initializer
init = tf.global_variables_initializer()

# Starting the tensorflow session
with tf.Session() as sess:
	# Initializing the Variables
	sess.run(init)
	cost_history, accuracy_history = [], []
	# Iterating through all the epochs
	for epoch in range(epochs):
		cost_per_epoch = 0
		sess.run(Optimizer, feed_dict={X : X_dataset, Y : Y_dataset})
		c = sess.run(Cost, feed_dict={X : X_dataset, Y : Y_dataset})
		correct_prediction = tf.equal(tf.argmax(Hypothesis, 1), tf.argmax(Y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		cost_history.append(sum(sum(c)))
		accuracy_history.append(accuracy.eval({X : X_dataset, Y : Y_dataset}) * 100)

		if epoch % 100 == 0 and epoch != 0:
			print("Epoch " + str(epoch) + " Cost: " + str(cost_history[-1]))
	# Storing the necessary values to be used outside the tensorflow session
	Weight = sess.run(W)
	Bias = sess.run(B)

	correct_prediction = tf.equal(tf.argmax(Hypothesis, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("\nAccuracy: ", accuracy_history[-1],"%")
#print("\n Final Training Cost = \n", c, "\n Final Weight = \n", Weight, "\n Final Bias = \n", Bias, '\n')

'''
# Plot the Cost history versus the Epochs
plt.plot(list(range(epochs)), cost_history)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Decrease in Cost with Epochs')
plt.show()
'''
'''
# Plot the Accuracy versus the Epochs
plt.plot(list(range(epochs)), accuracy_history)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Increase in Accuracy with Epochs')
plt.show()
'''


