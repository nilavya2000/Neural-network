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

#importing all the modules
import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    '''
    The sigmoid function, often used as an activation function in neural networks,
    especially in binary classification problems. It squashes the input values between 0 and 1,
    facilitating non-linear transformations in the network.
    '''
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    '''
    The derivative of the sigmoid function, which is important for the backpropagation algorithm
    in neural networks. It is used to calculate gradients for weights and biases updates,
    enabling the network to learn from the training data.
    '''
    return x * (1 - x)

# Initialize parameters
def initialize_parameters(input_size, hidden_size, output_size):
    '''
    Function to initialize weights & biases for a feedforward neural network with one hidden layer.
    
    input_size: The number of neurons in the input layer, which corresponds to the number of features in the dataset.
    hidden_size: The number of neurons in the hidden layer.
    output_size: The number of neurons in the output layer. For binary classification, this would typically be 1.
    '''
    np.random.seed(2)
    # Initialize weights between input layer and hidden layer
    weights_input_hidden = np.random.rand(input_size, hidden_size) - 0.5
    # Initialize weights between hidden layer and output layer
    weights_hidden_output = np.random.rand(hidden_size, output_size) - 0.5
    # Initialize bias for the hidden layer to zeros
    bias_hidden = np.zeros((1, hidden_size))
    # Initialize bias for the output layer to zeros
    bias_output = np.zeros((1, output_size))
    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

# Forward propagation
def forward_propagation(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    '''
    Function to perform forward propagation through the neural network.
    
    X: Input data
    weights_input_hidden: Weights between input layer and hidden layer
    weights_hidden_output: Weights between hidden layer and output layer
    bias_hidden: Bias for the hidden layer
    bias_output: Bias for the output layer
    '''
    # Calculate hidden layer input
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    # Apply sigmoid activation function to hidden layer input
    hidden_output = sigmoid(hidden_input)
    # Calculate output layer input
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    # Apply sigmoid activation function to output layer input
    final_output = sigmoid(final_input)
    return hidden_output, final_output

# Compute the loss
def compute_loss(Y, Y_hat):
    '''
    Function to compute the loss between predicted output and actual output.
    
    Y: Actual output
    Y_hat: Predicted output
    '''
    m = Y.shape[0]
    # Compute cross-entropy loss
    loss = -(1/m) * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
    return loss

# Backpropagation
def backpropagation(X, Y, hidden_output, final_output, weights_hidden_output, weights_input_hidden):
    '''
    Function to perform backpropagation to calculate gradients for weights and biases updates.
    
    X: Input data
    Y: Actual output
    hidden_output: Output of the hidden layer
    final_output: Predicted output
    weights_hidden_output: Weights between hidden layer and output layer
    weights_input_hidden: Weights between input layer and hidden layer
    '''
    # Compute error at output layer
    error = final_output - Y
    # Compute gradients for weights between hidden and output layers
    d_weights_hidden_output = np.dot(hidden_output.T, error * sigmoid_derivative(final_output))
    # Compute gradients for bias at output layer
    d_bias_output = np.sum(error * sigmoid_derivative(final_output), axis=0, keepdims=True)
    
    # Compute error at hidden layer
    error_hidden = np.dot(error * sigmoid_derivative(final_output), weights_hidden_output.T)
    # Compute gradients for weights between input and hidden layers
    d_weights_input_hidden = np.dot(X.T, error_hidden * sigmoid_derivative(hidden_output))
    # Compute gradients for bias at hidden layer
    d_bias_hidden = np.sum(error_hidden * sigmoid_derivative(hidden_output), axis=0, keepdims=True)
    
    return d_weights_input_hidden, d_weights_hidden_output, d_bias_hidden, d_bias_output

# Update parameters
def update_parameters(parameters, grads, learning_rate=1.0):
    '''
    Function to update weights and biases based on gradients computed during backpropagation.
    
    parameters: Tuple containing weights and biases
    grads: Tuple containing gradients for weights and biases
    learning_rate: Learning rate for gradient descent
    '''
    weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = parameters
    d_weights_input_hidden, d_weights_hidden_output, d_bias_hidden, d_bias_output = grads
    
    # Update weights between input and hidden layers
    weights_input_hidden -= learning_rate * d_weights_input_hidden
    # Update weights between hidden and output layers
    weights_hidden_output -= learning_rate * d_weights_hidden_output
    # Update bias at hidden layer
    bias_hidden -= learning_rate * d_bias_hidden
    # Update bias at output layer
    bias_output -= learning_rate * d_bias_output
    
    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

# Neural network training function
def train(X, Y, input_size, hidden_size, output_size, learning_rate, epochs):
    '''
    Function to train the neural network using gradient descent.
    
    X: Input data
    Y: Actual output
    input_size: Number of features in the input data
    hidden_size: Number of neurons in the hidden layer
    output_size: Number of neurons in the output layer
    learning_rate: Learning rate for gradient descent
    epochs: Number of iterations for training
    '''
    # Initialize parameters
    weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = initialize_parameters(input_size, hidden_size, output_size)
    
    for epoch in range(epochs):
        # Perform forward propagation
        hidden_output, final_output = forward_propagation(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
        # Compute loss
        loss = compute_loss(Y, final_output)
        # Perform backpropagation
        grads = backpropagation(X, Y, hidden_output, final_output, weights_hidden_output, weights_input_hidden)
        # Update parameters
        weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = update_parameters((weights_input_hidden, weights_hidden_output, bias_hidden, bias_output), grads, learning_rate)
        
        # Print loss every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

# Example usage
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.1
epochs = 10000