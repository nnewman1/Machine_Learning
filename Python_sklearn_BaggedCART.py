# Python tutorial using scikit-learn with Bagged Decision Trees on the pima indians diabetes dataset.
# Bootstrap Aggregation (Bagging) involves taking multiple samples from your whole training dataset and then training sub-models (typically of the same type) from each sample.
# The final output prediction is averaged across the predictions of all of the sub-models.
# Python is an interpreted, high-level, general-purpose programming language.
# sci-kit learn or sklearn is an high-level machine learning library for python.

#Import python libraries
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Import the Pima Indians Diabetes dataset
link = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# Create Features and Label names for the dataset
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# Create a Pandas Dataframe with the data's features and label names
theData = pd.read_csv(link, names=names)
# Analyze the DataFrame's head
#print("DataFrame's Head: \n", theData.head(), '\n')

# Create an array containing both the feature dataset and target dataset
theArray = theData.values
# Analyze the newly created array containing the dataset's values
#print("The Dataset Array: \n", theArray, '\n')
# Create the X dataset (Features) using the Array
X_Dataset = theArray[:,0:8]
# Create the Y dataset (labels) using the Array
Y_Dataset = theArray[:,8]

# Set the seed for reproducibility 
theSeed = 7
# Set the number of decision trees for the Bagging Classifier
num_trees = 100
# Using sklearn's K-Fold function from model_selection specify the number of folds to be used
kfold = model_selection.KFold(n_splits=10, random_state=theSeed)
# Create the CART Model that will be applied to the Bagging Classifier
theCART = DecisionTreeClassifier()
# Create the Bagging Classifier Model with the CART model
theModel = BaggingClassifier(base_estimator=theCART, n_estimators=num_trees, random_state=theSeed)
# Run the defined model and collect the results with the specified values
results = model_selection.cross_val_score(theModel, X_Dataset, Y_Dataset, cv=kfold)
# Analyze the models accuracy using the mean results
print("\n The Model's Accuracy Mean: ", results.mean(), "\n")
