# Python tutorial using pytorch for linear regression on a sudo generated dataset.
# Linear Regression is a common type of model for predictive analysis.
# The model is a linear approach to modeling the relationship between a scalar response (dependent variable) and explanatory variables (independent variable).
# Python is an interpreted, high-level, general-purpose programming language.
# Pytorch is an high-level machine learning library for python, based on the Torch library.
''' Linear Regression Model

y = X * beta + c + E

y = target
X = data
beta = coefficients
c = intercept
E = Error
'''

# import python libraries
import torch 
from torch.autograd import Variable 

# Create the randomly genereated linear datasets (X & Y)
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]])) 
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]])) 
  
# Define the LinearRegressionModel Class
class LinearRegressionModel(torch.nn.Module): 
  
    def __init__(self): 
        super(LinearRegressionModel, self).__init__() 
        self.linear = torch.nn.Linear(1, 1)  # One in and one out 
  
    def forward(self, x): 
        y_pred = self.linear(x) 
        return y_pred 
  
# Define the Linear regression model
our_model = LinearRegressionModel() 

# Define the Criterion (Loss) Function
criterion = torch.nn.MSELoss(size_average = False) 
# Define the Optimizer (SGD) Function
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.01) 

# Define the number of Epochs
for epoch in range(500): 
  
    # Forward pass: Compute predicted y by passing  
    # x to the model 
    pred_y = our_model(x_data) 
  
    # Compute and print loss 
    loss = criterion(pred_y, y_data) 
  
    # Zero gradients, perform a backward pass,  
    # and update the weights. 
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 
    # Analyze each epoch and corresponding loss value
    print('epoch {}, loss {}'.format(epoch, loss.data)) 

# Create a new variable to test the trained model
new_var = Variable(torch.Tensor([[4.0]]))
# Test the new variable against the trained model
pred_y = our_model(new_var)
# Analyze the new variable against the the trained model
print("predicted After Training: ", our_model(new_var).data[0][0])

