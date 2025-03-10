# nn\main.ipynb

This Jupyter notebook implements and tests a neural network for classifying handwritten digits from the MNIST dataset. Here's a breakdown of its key sections:

## Initial Setup and Testing
- Imports the custom NeuralNet class and numpy
- Creates a test neural network with 3 input, hidden, and output nodes
- Prints network parameters and test prediction

## Data Loading and Preparation 
- Loads MNIST training and test data from CSV files
- Converts data from strings to integers
- Separates images and their corresponding digit labels
- Scales pixel values to range [0.01, 1.0]
- Creates one-hot encoded target vectors

## Neural Network Creation
- Creates network with 784 input nodes (28x28 pixels), 100 hidden nodes, and 10 output nodes
- Defines accuracy calculation function
- Trains network for 10 epochs on training data

## Testing and Results
- Prepares test data with same scaling as training data
- Makes predictions on test set 
- Calculates and prints accuracy (~83%)

The notebook demonstrates the full pipeline of loading data, training a neural network, and evaluating its performance on a classic machine learning task. The achieved accuracy shows the network successfully learned to recognize handwritten digits.

[2 blank lines follow in original]