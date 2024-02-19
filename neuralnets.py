'''
Neural Net from Scratch

1. Define neural architecture
  - 2 i/p feature
  - 1 hidden layer of 4 neurons
  - 1 o/p neuron 
2. Initiate weights.
3. Forward propagation.
4. Calculate the Loss.
5. Backward Propagation.
6. Update parameters.
'''

#importing the modules
import numpy as np

#sigmoid function
def sigmoid(x):
  '''
  sigmoid function, often used as an activation function in neural networks, 
  especially in binary classification problems.
  '''
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  '''
  is important for the back propagation algorithm in neural networks, 
  as it is used to calculate gradients for weights and biases updates.
  '''
  return x * (1 - x)
