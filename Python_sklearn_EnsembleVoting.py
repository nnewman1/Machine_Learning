# Python tutorial using scikit-learn with ensemble voting for classification on pima indians diabetes
# Ensemble voting is building multiple models (typically of differing types) and then simple statistics (like calculating the mean) are used to combine predictions
# Python is an interpreted, high-level, general-purpose programming language.
# sci-kit learn or sklearn is an high-level machine learning library for python.

# Import python libraries
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

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
# Set the number of decision trees the ensemble voting classifier
num_trees = 100
# Using sklearn's K-Fold function from model_selection specify the number of folds to be used
kfold = model_selection.KFold(n_splits=10, random_state=theSeed)

# Create the container holding the model's estimators
theEstimators = []

# Create the Logistic Regression model
theLRmodel = LogisticRegression()
# Append the Logistic Regression model to the estimator's container 
theEstimators.append(('LogisticRegression', theLRmodel))

# Create the Decision Tree Classifier model
theDTmodel = DecisionTreeClassifier()
# Append the Decision Tree model to the estimator's container
theEstimators.append(('DecisionTree', theDTmodel))

# Create the Support Vector Machine model
theSVMmodel = SVC()
# Append the Support Vector Machine model to the estimator's container
theEstimators.append(('SupportVectorMachine', theSVMmodel))

# Create the ensemble voting model with all three different models
theEnsemble = VotingClassifier(theEstimators)
# Run the ensemble model and collect the results with the specified values
theResults = model_selection.cross_val_score(theEnsemble, X_Dataset, Y_Dataset, cv=kfold)

# Analyze each model's sepcifications
#print("\n Each Model's specifictions: ", theEnsemble, "\n")
# Analyze the Ensemble Voting model's accuracy using using the mean results
print("\nThe Model's Accuracy Mean: ", theResults.mean(), "\n")

