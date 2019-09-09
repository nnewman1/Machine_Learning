# Python tutorial using scikit-learn for the perceptron model on the iris dataset.
# A perceptron is a simple model of a biological neuron in an artificial neural network.
# The perceptron algorithm classifies input data by finding the linear separation between different objects and patterns that are received through numeric or visual input.

# Import python libraries
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import the iris dataset
dataSet = datasets.load_iris()

# Analyze the feature set of the data
#print("features: ", dataSet.feature_names, "\n")

# Analyze the target set of the data
#print("Labels: ", dataSet.target, "\n")

# Analyze the dataset's shape 
#print("Data's Shape: ", dataSet.data.shape, "\n")

# Analyze the first five entires of the dataset's values
#print("Data's First Five: ", dataSet.data[0:5], "\n")

# Split the whole dataset into a seperate training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(dataSet.data, dataSet.target, test_size=0.3)

# Transform the data into standardization form for better classication results
theScaler = StandardScaler()
# Train the Scaler to the X_train dataset
theScaler.fit(X_train)

# Transform the X_train dataset with the Scaler
X_train_std = theScaler.transform(X_train)
# Transform the X_test dataset with the Scaler
X_test_std = theScaler.transform(X_test)

# Create the Perceptron Model with 60 iterations and 0.15 learning rate
theModel = Perceptron(max_iter=60, eta0=0.15, random_state=0)
# Train the newly created Perceptron model using both the X_train_std and Y_train datasets
theModel.fit(X_train_std, y_train)

# Predict the testing dataset using the recently trained Perceptron model
theModel_Predict = theModel.predict(X_test_std)

# Analyze the Perceptron model's Accuracy, confusion_matrix, and classification report
print("The Model's Accuracy: ", accuracy_score(y_test, theModel_Predict), "\n")
print("The Model's Confusion Matrix: \n", confusion_matrix(y_test, theModel_Predict), "\n")
print("The Model's Classicication Report: \n", classification_report(y_test, theModel_Predict), "\n")
