# nn/main.ipynb Documentation

This Jupyter notebook implements and tests a neural network for digit recognition using the MNIST dataset. Here's a breakdown of the key sections:

## Initial Setup and Test
- Imports the custom NeuralNet class and numpy
- Creates and tests a small neural network with 3 nodes in each layer
- Prints network object and weight matrices

## Data Loading and Preparation
- Loads MNIST training and test data from CSV files
- Converts image data from strings to integers
- Separates labels (digits) from pixel data
- Scales pixel values to range 0.01-1.0
- Creates one-hot encoded target vectors

## Neural Network Creation
- Creates network with:
  - 784 input nodes (28x28 pixels)
  - 100 hidden nodes
  - 10 output nodes (digits 0-9)
- Implements accuracy calculation function

## Training
- Trains network for 10 epochs
- For each epoch:
  - Processes each training image/target pair
  - Updates network weights via backpropagation

## Testing and Results
- Prepares test data using same scaling as training data  
- Makes predictions on test dataset
- Compares predictions to actual labels
- Calculates and prints accuracy (~83%)

The notebook demonstrates a complete workflow of training and testing a neural network on the MNIST digit recognition task, achieving reasonable accuracy for a simple implementation.# nn/neural_net.py

This file implements a basic three-layer neural network using NumPy and SciPy.

## Class: NeuralNet

A neural network implementation with one hidden layer, using sigmoid activation function.

### Constructor Parameters
- `input_n`: Number of input nodes
- `output_n`: Number of output nodes
- `hidden_n`: Number of hidden nodes
- `learning_rate`: Learning rate for weight updates
- `internal_layers`: Unused parameter (appears to be for future expansion)
- `internal_nc`: Unused parameter (appears to be for future expansion)

### Attributes
- `wh_i`: Weight matrix between input and hidden layer
- `wh_o`: Weight matrix between hidden and output layer
- `activation_function`: Sigmoid activation function using scipy.special.expit

### Methods

#### train(inputs, targets)
Trains the neural network using backpropagation.

**Parameters:**
- `inputs`: Input data for training
- `targets`: Expected output (labels)

**Process:**
1. Forward propagation through the network
2. Calculate errors at output and hidden layers
3. Update weights using gradient descent

#### predict(inputs)
Makes predictions using the trained network.

**Parameters:**
- `inputs`: Input data for prediction

**Returns:**
- Final output after forward propagation through the network

## Technical Details
- Uses NumPy for matrix operations
- Implements sigmoid activation function
- Weights are initialized using normalized random values
- Supports batch processing through NumPy array operations
