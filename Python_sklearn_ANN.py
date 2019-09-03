# Python tutorial using scikit-learn with a one hidden layer Artificial Neural Network (ANN) on the iris dataset.
# An Artificial Neural Network is based on the structure of a biological brain. 
# These systems learn to perform tasks or classify based on data, without the need to be programmed specific task rules.

# Import python libraries
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import the iris dataset
dataSet = datasets.load_iris()

# Analyze the feature set of the data
#print("features: ", dataSet.feature_names, "\n")

# Analyze the target set of the data
#print("Labels: ", dataSet.target_names, "\n")

# Analyze the dataset's shape 
#print("Data's Shape: ", dataSet.data.shape, "\n")

# Analyze the first five entires of the dataset's values
#print("Data's First Five Values: ", dataSet.data[0:5], "\n")

# Analyze the target set of the data
#print("Data's Target Values: ", dataSet.target, "\n")

# Split the whole dataset into a seperate training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(dataSet.data, dataSet.target, test_size=0.20)

# Create a Scaler to transform data into standardization form for better classication results
theScaler = StandardScaler()
# Train the Scaler to the X_train dataset
theScaler.fit(X_train)

# Transform the X_train dataset with the Scaler
X_train = theScaler.transform(X_train)
# Transform the X_test dataset with the Scaler
X_test = theScaler.transform(X_test)

# Create the ANN Model with one hidden layer that has ten nodes and 1000 iterations and default learning rate
theModel = MLPClassifier(activation='relu', hidden_layer_sizes=(10), max_iter=1000)
# Train the newly created ANN Model using both X_Train and Y_Train datasets
theModel.fit(X_train, y_train)

# Predict the testing dataset using the recently trained ANN model
theModel_Predict = theModel.predict(X_test)

# Analyze the ANN model's Accuracy, confusion_matrix, and classification report
print("The Model's Accuracy: ", accuracy_score(y_test, theModel_Predict), "\n")
print("The Model's Confusion Matrix: \n", confusion_matrix(y_test, theModel_Predict), "\n")
print("The Model's Classicication Report: \n", classification_report(y_test, theModel_Predict), "\n")

# Analyze the ANN model's weight matrices and Bias vectors
print("The Models Weight Matrices: \n", theModel.coefs_, "\n")
print("The Models Bias Vectors: \n", theModel.intercepts_, "\n")

