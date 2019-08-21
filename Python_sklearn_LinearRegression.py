# Python tutorial using scikit-learn for linear regression on diabetes dataset.
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

# Import python libraries
import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Import the diabetes dataset
dataSet = datasets.load_diabetes()

# Analyze the feature set of the data
#print("features: ", dataSet.feature_names)

# Analyze the target set of the data
#print("Labels: ", dataSet.target_names)

# Analyze the dataset's shape 
#print("Data's Shape: ", dataSet.data.shape)

# Analyze the target set's shape of the data
#print("Data's Target Shape: ", dataSet.target.shape)

# Analyze the first five entires of the dataset's values
#print("Data's First Five: ", dataSet.data[0:5])

# Analyze the target set of the data
#print("Data's Target: ", dataSet.target)

# Split the whole dataset into a seperate training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(dataSet.data, dataSet.target, test_size=0.2, random_state=0)

# Create the linear regression model
theModel = LinearRegression()
# Train the model using the training sets
theModel.fit(X_train, y_train)

# Predict the testing dataset using the recently created Linear regression model.
theModel_Predict = theModel.predict(X_test)

# Analyze the accuracy score of the model using the testing dataset
score = theModel.score(X_test, y_test)
print("Accuracy: ", score)

# Analyze the model's Coefficients
theModel_Coefficients = theModel.coef_
print("Coefficients: ", theModel_Coefficients)

# Analyze the model's Intercept
theModel_Intercept = theModel.intercept_
print("Intercept: ", theModel_Intercept)

# Analyze the Linear regression model's results using a plot
plt.plot(y_test, theModel_Predict, '.')
x = np.linspace(0, 330, 100)
y = x
plt.plot(x, y)
plt.show()

