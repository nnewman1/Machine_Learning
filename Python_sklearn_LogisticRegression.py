# Python tutorial using scikit-learn for Logistic Regression on the Digits dataset.
# The logistic regression model is regression analysis when the dependent variable is binary (0 or 1).
# Python is an interpreted, high-level, general-purpose programming language.
# sci-kit learn or sklearn is an high-level machine learning library for python.

# Import python libraries
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Import the digits dataset
dataSet = load_digits()

# Analyze the dataset's shape
#print("Dataset Shape: ", dataSet.data.shape)

# Analyze the target set of the data
#print("Dataset Labels: ", dataSet.target)

# Analyze the images for the first 5 digit in the dataset using matplotlib
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(dataSet.data[0:5], dataSet.target[0:5])):
	plt.subplot(1, 5, index + 1)
	plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
	plt.title('training: %i' % label, fontsize=20)
#plt.show()

# Split the whole dataset into a seperate training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(dataSet.data, dataSet.target, test_size=0.25, random_state=0)

# Create the Logistic Regression model 
theModel = LogisticRegression()

# Train the model using the training datasets
theModel.fit(X_train, y_train)

# Predict the testing dataset using the recently created logistic regression model
theModel_Predict = theModel.predict(X_test)

# Score the prediction of the trained model
score = theModel.score(X_test, y_test)

# Analyze the model's accuracy score
print("Model's Accuracy Score: ", score)

# Using the sklearn metrics library analyze the models confusion matrix
cm = metrics.confusion_matrix(y_test, theModel_Predict)
print('\n',cm)

