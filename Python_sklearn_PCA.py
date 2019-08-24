# Python tutorial using scikit-learn for Principal Component Analysis (PCA) on the digits dataset.
# Principal component analysis (PCA) is used to explain the variance-covariance structure of a set of variables through linear combinations. 
# It is often used as a dimensionality-reduction technique.

# Import python libraries
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# load the digits dataset
theData = load_digits()

# Analyze the dataset's shape
print("The Data's shape: ", theData.data.shape, "\n")

# Create the PCA model and specify the number of components 
theModel = PCA(n_components=2)

# Train the PCA model using the X data
Transformed = theModel.fit_transform(theData.data)

# Analyze the dataset before the PCA transformation
print("Before PCA transformation: ", theData.data.shape, "\n")

# Analyze the dataset after the PCA transformation
print("After PCA transformtion: ", Transformed.shape, "\n")

# Analyze the PCA's component values
print("The PCA's components: ", theModel.components_, "\n")

# Analyze the PCA's explained variance
print("The PCA's Explained Variance: ", theModel.explained_variance_, "\n")

# Analyze the PCA's explained variance ratio
print("the PCA's Explained Variance Ratio: ", theModel.explained_variance_ratio_, "\n")

# Create a scatter plot showing the data after the PCA transformation 
plt.scatter(Transformed[:, 0], Transformed[:, 1], c=theData.target, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral_r', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();
#plt.show()

####################  Number of Components Selection ####################

# Create and train a new PCA model with the full dataset to determine the optimal number of components to correctly express the data and plot the results
pca = PCA().fit(theData.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
#plt.show()
