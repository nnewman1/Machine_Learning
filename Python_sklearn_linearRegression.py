# Python tutorial using scikit-learn for linear regression on diabetes dataset.

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets

dataSet = datasets.load_diabetes()

#print(dataSet)
#print(dataSet.data.shape) #(422, 10)
#print(dataSet.target.shape) #(442, )
#print(dataSet.feature_names) #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

''' Linear Regression Model

y = X * beta + c + E

y = target
X = data
beta = coefficients
c = intercept
E = Error
'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataSet.data, dataSet.target, test_size=0.2, random_state=0)

theModel = LinearRegression()
theModel.fit(X_train, y_train)
score = theModel.score(X_test, y_test)
#print(score)

theModel_Coefficients = theModel.coef_
#print(theModel_Coefficients)

theModel_Intercept = theModel.intercept_
#print(theModel_Intercept)

theModel_Predict = theModel.predict(X_test)
#print(theModel_Predict)

plt.plot(y_test, theModel_Predict, '.')
x = np.linspace(0, 330, 100)
y = x
plt.plot(x, y)
plt.show()




