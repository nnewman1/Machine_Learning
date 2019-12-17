# Python tutorial using scikit-learn for K-Nearest Neighbors (KNN) on wine dataset.
# K-Nearest Neighbors is a simple classification algorithm that uses a dataset, in which the data points are separated into several classes to predict the classification of a new sample point.
# Python is an interpreted, high-level, general-purpose programming language.
# sci-kit learn or sklearn is an high-level machine learning library for python.

# Import python libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Import the wine dataset
dataSet = datasets.load_wine()

# Analyze the feature set of the data
#print("features: ", dataSet.feature_names)

# Analyze the target set of the data
#print("Labels: ", dataSet.target_names)

# Analyze the first five entires of the dataset's values
#print("Data's First Five: ", dataSet.data[0:5])

# Analyze the target set of the data
#print("Data's Target: ", dataSet.target)

# Analyze the dataset's shape 
#print("Data's Shape: ", dataSet.data.shape)

# Split the whole dataset into seperate training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(dataSet.data, dataSet.target, test_size=0.3)

# Create the KNN model
theModel = KNeighborsClassifier(n_neighbors=5)
# Train the model using the training sets
theModel.fit(X_train, y_train)
# Predict the testing dataset using the recently created KNN model
theModel_Predict = theModel.predict(X_test)

# Using the sklearn metrics library analyze the models accuracy
print("Accuracy: ", metrics.accuracy_score(y_test, theModel_Predict))

