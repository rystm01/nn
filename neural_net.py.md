# nn/neural_net.py

This file implements a basic three-layer neural network using NumPy and SciPy.

## Class: NeuralNet

A implementation of a three-layer neural network with configurable input, hidden, and output layers.

### Constructor Parameters
- `input_n`: Number of input nodes
- `output_n`: Number of output nodes
- `hidden_n`: Number of hidden nodes
- `learning_rate`: Learning rate for weight updates
- `internal_layers`: (Unused) Number of internal layers
- `internal_nc`: (Unused) List for internal layer configuration

### Attributes
- `wh_i`: Weight matrix between input and hidden layer
- `wh_o`: Weight matrix between hidden and output layer
- `activation_function`: Sigmoid activation function using scipy.special.expit

### Methods

#### train(inputs, targets)
Trains the neural network using backpropagation.

**Parameters:**
- `inputs`: Input data
- `targets`: Target/expected output data

**Process:**
1. Forward propagation through the network
2. Calculate errors at output and hidden layers
3. Update weights using gradient descent

#### predict(inputs)
Makes predictions using the trained network.

**Parameters:**
- `inputs`: Input data to make predictions on

**Returns:**
- Final output predictions after forward propagation

### Implementation Details
- Uses NumPy for matrix operations
- Implements sigmoid activation function
- Initializes weights using normal distribution
- Uses gradient descent for learning
- Supports single hidden layer architecture

