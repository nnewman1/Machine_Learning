# Python tutorial using scikit-learn for random forest classification using the iris dataset.
# The Random Forest algorithm is an extension of bagged decision trees (CART).
# Python is an interpreted, high-level, general-purpose programming language.
# sci-kit learn or sklearn is an high-level machine learning library for python.
# Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy.
# Seaborn is a Python data visualization library based on matplotlib.

# Import python libraries
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Import the iris dataset
dataSet = datasets.load_iris()

# Analyze the feature set of the data
#print("Feature Names: ", dataSet.feature_names, "\n")

# Analyze the target names of the data
#print("Label Names: ", dataSet.target_names, "\n")

# Analyze the first five entires of the dataset's values
#print("Data's First Five Records: ", dataSet.data[0:5], "\n")

# Analyze the target set of the data
#print("Data's Target Values: ", dataSet.target, "\n")

# Create an Pandas DataFrame with the data's features and labels
theData = pd.DataFrame({'sepal length':dataSet.data[:,0], 'sepal width':dataSet.data[:,1], 'petal length':dataSet.data[:,2], 'petal width':dataSet.data[:,3], 'species':dataSet.target})

# Analyze the DataFrame's head
#print("DataFrame's Head: \n", theData.head(), '\n')

# Create the X dataset (Features) using the dataframe
X_dataSet = theData[['sepal length', 'sepal width', 'petal length', 'petal width']]

# Create the Y dataset (labels) using the dataframe
Y_dataSet = theData['species']

# Split the whole dataset into a seperate training and testing dataset (Pandas DataFrame Method)
X_train, X_test, y_train, y_test = train_test_split(X_dataSet, Y_dataSet, test_size=0.3)
# Split the whole dataset into a seperate training and testing dataset (dataSet Method)
#X_train, X_test, y_train, y_test = train_test_split(dataSet.data, dataSet.target, test_size=0.2, random_state=0)

# Create the Random Forest model
theModel = RandomForestClassifier(n_estimators=100)

# Train the Random Forest model with the X and Y training datasets
theModel.fit(X_train, y_train)

# Predict the testing dataset using the recently created Random Forest model
theModel_Predict = theModel.predict(X_test)
# Analyze the model's accuracy using the whole dataset
print("Model's Accuracy: ", metrics.accuracy_score(y_test, theModel_Predict), '\n')

# Predict a single entry using the recently created Random Forest model
#theModel_Predict = theModel.predict([[3, 4, 4, 2]])
# Analyze the model's accuracy on the single prediction
#print("Single Prediction: ", theModel_Predict, "\n")

#################### Feature Importance Improvement #####################

# Generate a dataframe gathering the most important features of the dataset
feature_imp = pd.Series(theModel.feature_importances_,index=dataSet.feature_names).sort_values(ascending=False)

# Analyze the feature importance dataframe
#print("feature Importance Values:\n", feature_imp, "\n")

# Creating a bar plot of the feature importance using matplotlib and seaborn
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to the graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
#plt.show()
