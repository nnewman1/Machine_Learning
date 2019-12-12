# Python tutorial using scikit-learn on dataset preprocessing for machine learning algorithms.
# Data Preprocessing is the most crucial step of the machine learning and deep learning workflow pipeline.
# Python is an interpreted, high-level, general-purpose programming language.
# sci-kit learn or sklearn is an high-level machine learning library for python.

# Import python libraries
from sklearn import preprocessing 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
import pandas
import scipy
import numpy

########################## Create working dataset workflow ##########################

# Input data from url link
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# List names to be applied to the data columns 
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# Turn data from url and column names to create an pandas dataframe
dataframe = pandas.read_csv(url, names=names)
# Set the dataframe's values into an array
array = dataframe.values
# separate the dataset array into X and Y arrays 
X = array[:,0:8]
Y = array[:,8]

########################## Rescale Dataset workflow ##########################

# Create rescale scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Transform the X data with the rescale scaler
rescaledX = scaler.fit_transform(X)

# Set X data values to the 3rd precision
numpy.set_printoptions(precision=3)

# Analyze the transformed X data
print("The Transformed Rescaled Data: ", rescaledX[0:5,:], '\n')

########################## Standardize Dataset workflow ##########################

# Create standardize scaler
scaler = StandardScaler()

# Transform the X data with the standardize scaler
rescaledX = scaler.fit_transform(X)

# Transform the X data with the scale function (2nd method)
#rescaledX = preprocessing.scale(X)

# Set X data values to the 3rd precision
numpy.set_printoptions(precision=3)

# Analyze the transformed X data
#print("The Transformed Standardized Data: ", rescaledX[0:5,:], '\n')

########################## Normalize Dataset workflow ##########################

# Create normalize scaler
scaler = Normalizer()

# Transform the X data with the normalize scaler
rescaledX = scaler.fit_transform(X)

# Transform the X data with the normalize function (2nd method)
#rescaledX = preprocessing.normalize(X)

# Set X data values to the 3rd precision
numpy.set_printoptions(precision=3)

# Analyze the transformed X data
#print("The Transformed Normalized Data: ", rescaledX[0:5,:], '\n')

########################## Binarize Dataset workflow ##########################

# Create binarizer scaler
binarizer = Binarizer(threshold=0.0)

# Transform the X data with the binarizer scaler
binaryX = binarizer.fit_transform(X)

# Set X data values to the 3rd precision
numpy.set_printoptions(precision=3)

# Analyze the transformed X data
#print("The Transformed Binarized Data: ", binaryX[0:5,:], '\n')

########################## Encoding Features and Labels workflow ##########################

# Create 1st X features of the dataset
Weather = ['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy']

# Create the 2nd X features of the dataset
Temp = ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

# Create the Y labels of the dataset
Play = ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

# Create the Encoder using the scikit-learn preprocessing function
le = preprocessing.LabelEncoder()

# Transform the 1st feature using the encoder
weather_encoded = le.fit_transform(Weather)
# Analyze the transformed X data
print("The Transformed Weather Data: ", weather_encoded, '\n')
print("The Orginal Weather Data", Weather, '\n')

# Transform the 2nd feature using the encoder
temp_encoded = le.fit_transform(Temp)
# Analyze the transformed X data
print("The Transformed Temp Data: ", temp_encoded, '\n')
print("The Orginal Temp Data", Temp, '\n')

# Transform the Y labels using the encoder
play_encoded = le.fit_transform(Play)
# Analyze the transformed Y labels
print("The Transformed Play Data: ", play_encoded, '\n')
print("The Orginal Play Data", Play, '\n')

# Create a tuple with both the 1st and 2nd feature to create a full feature dataset
features = zip(weather_encoded, temp_encoded)
#Analyze the full feature dataset by casting the tuple to a list
print("The Transformed Features (X) Data: ", list(features), '\n')



