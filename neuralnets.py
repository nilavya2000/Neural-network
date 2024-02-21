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

#initialize parameters
def initialize_parameters(input_size, hidden_size, output_size):
  '''
  Function to initialize weights & Biases for a feedforward neural nets with one hidden layer.
  input_size: The number of neurons in the input layer, which corresponds to the number of features in the dataset.
  hidden_size: The number of neurons in the hidden layer.
  output_size: The number of neurons in the output layer. For binary classification, this would typically be 1.
  '''
  np.random.seed(2) #it ensures that the random numbers generated will be same every time the code is run.
  weights_input_hidden = np.random.rand(input_size, hidden_size) - 0.5 #this initialize the weights between the i/p layer and the hidden layer. and 0.5 is subtracted so that the values remain in -0.5 to 0.5.
  weights_hidden_output = np.random.rand(hidden_size, output_size) - 0.5 #this initialize the weights between the hidden layer and the output layer. and 0.5 is subtracted so that the values remain in -0.5 to 0.5.
  bias_hidden= np.zeros((1, hidden_size)) # the bias for the hidden layer to 0's. It means that there is only one bias term per neuron in the hidden layer.
  bias_output= np.zeros((1, output_size)) # the bias for the output layer to 0's. It means that there is only one bias term per neuron in the hidden layer.
  return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

def forward_propagation(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
  '''
  Function is responsible for performing the forward pass of a neural net. It takes i/p X and moves through the network to get the o/p using weights bias & activated Function.
  '''
  hidden_input = np.dot(X, weights_input_hidden) + bias_hidden # this provides a i/p to the hidden layer where there is a matrix multiplication between the i/p X and weights and then it added some bias to it.
  hidden_output = sigmoid(hidden_input) #activating the activation function. so this is the activation function for each neuron in the hidden layer.
  final_input = np.dot(hidden_output, weights_hidden_output) + bias_output #similarly as hidden i/p.
  final_output = sigmoid(final_input) #activating the activation function.

  return hidden_output, final_output

