# Python tutorial using scikit-learn for support vector machine (SVM) on breast cancer dataset
''' SVM Kernels

Linear Kernel = sum(x * xi)
Polynomial Kernel = 1 + sum(x * xi) ^ d
Radial Basis Function Kernel = exp(-gamma * sum((x - xi) ^ 2)
'''
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

# Import the breast cancer dataset
dataSet = datasets.load_breast_cancer()

# Analyze the feature set of the data
#print("features: ", dataSet.feature_names)

# Analyze the target set of the data
#print("Labels: ", dataSet.target_names)

# Analyze the dataset's shape 
#print(dataSet.data.shape)

# Analyze the first five entires of the dataset's values
#print(dataSet.data[0:5])

# Analyze the target set of the data
#print(dataSet.target)

# Split the whole dataset into a seperate training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(dataSet.data, dataSet.target, test_size=0.3, random_state=109)

# Create the SVM model using the Linear Kernel
theModel = svm.SVC(kernel='linear')
# Train the model using the training sets
theModel.fit(X_train, y_train)

# Predict the testing dataset using the recently created SVM model.
theModel_Predict = theModel.predict(X_test)

# Using the sklearn metrics library analyze the models accuracy, precison, and recall
print("Accuracy: ", metrics.accuracy_score(y_test, theModel_Predict))
print("Precision: ", metrics.precision_score(y_test, theModel_Predict))
print("Recall: ", metrics.recall_score(y_test, theModel_Predict))

