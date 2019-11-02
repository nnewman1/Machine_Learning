# Python tutorial using scikit-learn for Naive Bayes Classifier on the wine dataset.
# The Naive Bayes Algorithm is based off of probabilistic classifiers by applying the Bayes' theorem with strong independence assumptions between the features.
# sci-kit learn or sklearn is an high-level machine learning library for python.
''' Naive Bayes Model

P(h|D) = P(D|h)P(h) / P(D)

P(h|D) = the probability of hypothesis h given the data D
P(D|h) = the probability of data D given the hypothesis h was true
P(h) = the probability of hypothesis h being true (regardless of the data)
p(D) = the probability of the data (regardless of the hypothesis)
'''

# Import python libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# Import the wine dataset
dataSet = datasets.load_wine()

# Analyze the feature set of the data
#print("Features: ", dataSet.feature_names)

# Analyze the target set of the data
#print("Labels: ", dataSet.target_names)

# Analyze the dataset's shape 
#print("Data's Shape: ", dataSet.data.shape)

# Analyze the first five entires of the dataset's values
#print("Data's First Five: ", dataSet.data[0:5])

# Analyze the target set of the data
#print("Data's Target: ", dataSet.target)

# Split the whole dataset into a seperate training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(dataSet.data, dataSet.target, test_size=0.3, random_state=109)

# Create the Naive Bayes Model
theModel = GaussianNB()

# Train the newly created Naive Bayes Model using both the X and y training datasets
theModel.fit(X_train, y_train)

# Predict the testing dataset using the recently trained Naive Bayes model
theModel_Predict = theModel.predict(X_test)

# Analyze the Naive Bayes model's Accuracy results
print("Accuracy: ", metrics.accuracy_score(y_test, theModel_Predict))

