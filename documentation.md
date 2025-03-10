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
nn\main.ipynb Documentation:

This notebook implements and tests a neural network for handwritten digit recognition using the MNIST dataset. Here's a section-by-section breakdown:

1. Initial Setup & Testing
- Imports neural network class and numpy
- Creates test neural network instance
- Tests basic prediction functionality

2. Dataset Loading
- Loads MNIST training and test data from CSV files
- Splits data into training and test sets

3. Data Preparation
- Converts data from strings to integers
- Separates images from their digit labels
- Scales pixel values to range 0.01-0.99
- Creates one-hot encoded target vectors

4. Neural Network Creation & Training
- Creates neural network with dimensions:
  - Input: 784 nodes (28x28 pixels)
  - Hidden: 100 nodes
  - Output: 10 nodes (digits 0-9)
- Trains network for 10 epochs on training data

5. Testing & Accuracy
- Preprocesses test data similarly to training data
- Makes predictions on test images
- Calculates accuracy by comparing predictions to actual digits
- Achieves approximately 82.94% accuracy on test set

6. Helper Functions
- Implements accuracy calculation function
- Contains data preprocessing functions

The notebook demonstrates a complete machine learning pipeline from data loading through training and evaluation, achieving reasonable accuracy on the MNIST digit recognition task.# nn/neural_net.py

This file implements a basic three-layer neural network using NumPy and SciPy.

## Class: NeuralNet

A neural network implementation with one hidden layer.

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
- `targets`: Target values

**Process:**
1. Forward propagation through hidden layer
2. Calculation of output predictions
3. Error calculation
4. Weight adjustment using gradient descent

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
- Uses basic backpropagation for training
# nn\main.ipynb - Documentation

This Jupyter notebook demonstrates the implementation and training of a neural network for digit recognition using the MNIST dataset. Here's a breakdown of the key sections:

## Initial Setup
- Imports the custom NeuralNet class and numpy
- Creates an initial test neural network with 3 neurons in each layer
- Tests basic prediction functionality

## Dataset Loading
- Loads MNIST training and test data from CSV files
- Training data loaded from 'data/mnist_train.csv'
- Test data loaded from 'data/mnist_test.csv'

## Data Preparation
- Converts image data from strings to integers
- Separates images and their corresponding digit labels
- Scales pixel values to range 0.01-1.0
- Creates one-hot encoded target vectors

## Neural Network Creation & Training
- Creates neural network with:
  - 784 input neurons (28x28 pixels)
  - 100 hidden neurons
  - 10 output neurons (digits 0-9)
  - Learning rate of 1
- Trains for 10 epochs on the full training dataset

## Testing & Evaluation
- Defines accuracy calculation function
- Prepares test data using same scaling as training data
- Makes predictions on test dataset
- Evaluates accuracy by comparing predictions to actual digits
- Achieves approximately 82.94% accuracy on test set

This implementation shows a basic but functional neural network recognizing handwritten digits, demonstrating core concepts of neural network training and evaluation.

# nn/neural_net.py

This file implements a basic three-layer neural network using NumPy and SciPy.

## Class: NeuralNet

A neural network implementation with one hidden layer, using sigmoid activation function.

### Constructor Parameters
- `input_n`: Number of input nodes
- `output_n`: Number of output nodes
- `hidden_n`: Number of hidden nodes
- `learning_rate`: Learning rate for weight updates
- `internal_layers`: (Unused) Number of internal layers
- `internal_nc`: (Unused) List of internal node counts

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
2. Calculate errors for output and hidden layers
3. Update weights using gradient descent

#### predict(inputs)
Makes predictions using the trained network.

**Parameters:**
- `inputs`: Input data to make predictions on

**Returns:**
- Final output predictions after forward propagation

### Implementation Details
- Uses NumPy for matrix operations
- Initializes weights using normal distribution
- Applies sigmoid activation function
- Implements basic gradient descent for learning

# nn\main.ipynb

This Jupyter notebook implements a neural network for digit recognition using the MNIST dataset. Here's a breakdown of the key sections:

1. **Initial Setup and Testing**
- Imports the NeuralNet class and numpy
- Creates a test neural network instance
- Performs initial prediction test

2. **Dataset Loading**
- Loads MNIST training and test data from CSV files
- Processes the data into appropriate formats

3. **Data Preparation**
- Converts image strings to integers
- Scales pixel values to appropriate range (0.01-0.99)
- Separates digits and images
- Creates target vectors for training

4. **Neural Network Creation**
- Configures network with:
  - 784 input neurons (28x28 pixels)
  - 100 hidden neurons
  - 10 output neurons (digits 0-9)

5. **Model Training**
- Implements accuracy calculation function
- Trains network for 10 epochs
- Uses batch training approach

6. **Testing and Evaluation**
- Prepares test data
- Makes predictions on test set
- Calculates and prints accuracy score (~82.94%)

The notebook demonstrates a complete workflow from data loading through training and evaluation of a basic neural network for digit recognition.# nn/neural_net.py

This file implements a three-layer neural network using NumPy and SciPy. 

## Class: NeuralNet

A neural network implementation with one hidden layer.

### Constructor Parameters
- `input_n`: Number of input nodes
- `output_n`: Number of output nodes
- `hidden_n`: Number of hidden nodes
- `learning_rate`: Learning rate for weight updates
- `internal_layers`: Unused parameter (default=1)
- `internal_nc`: Unused parameter (default=[])

### Attributes
- `wh_i`: Weight matrix between input and hidden layer
- `wh_o`: Weight matrix between hidden and output layer
- `activation_function`: Sigmoid activation function using scipy.special.expit

### Methods

#### train(inputs, targets)
Trains the neural network using backpropagation.

**Parameters:**
- `inputs`: Input data array
- `targets`: Target values array

**Process:**
1. Forward propagation through the network
2. Calculate errors at output and hidden layers
3. Update weights using gradient descent

#### predict(inputs)
Makes predictions using the trained network.

**Parameters:**
- `inputs`: Input data array

**Returns:**
- Final output values after forward propagation through the network

### Implementation Details
- Uses numpy for matrix operations
- Initializes weights using normal distribution
- Employs sigmoid activation function
- Implements basic backpropagation algorithm
- Single hidden layer architecture
# nn\main.ipynb

This Jupyter notebook demonstrates the implementation and training of a neural network for digit recognition using the MNIST dataset. Here's a section-by-section breakdown:

## Initial Setup and Testing
- Imports the custom NeuralNet class and numpy
- Creates a test neural network with dimensions 3x3x3x1
- Performs initial prediction test

## Dataset Loading
- Loads MNIST training and test data from CSV files
- Splits data into test and training sets

## Data Preparation 
- Converts image data from strings to integers
- Separates images and their corresponding digit labels
- Scales pixel values to range 0.01-1.0
- Creates one-hot encoded target vectors

## Neural Network Creation
- Configures network with:
  - 784 input neurons (28x28 pixels)
  - 100 hidden neurons
  - 10 output neurons (digits 0-9)
- Implements accuracy calculation function

## Training
- Trains network for 10 epochs
- Processes each image in training set

## Testing and Evaluation
- Prepares test data with same scaling as training data
- Makes predictions on test set
- Calculates and displays accuracy (~82.94%)

The notebook demonstrates a working implementation of a neural network achieving reasonable accuracy on the MNIST digit recognition task.

The code includes extensive error handling, data preprocessing, and evaluation metrics, making it a complete example of neural network implementation for image classification.# nn/neural_net.py

This file implements a basic three-layer neural network using NumPy and SciPy.

## Class: NeuralNet

A neural network implementation with input, hidden, and output layers.

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
- `targets`: Target/expected outputs

**Process:**
1. Forward propagation through the network
2. Calculate errors
3. Update weights using gradient descent

#### predict(inputs)
Makes predictions using the trained network.

**Parameters:**
- `inputs`: Input data to predict

**Returns:**
- Final output predictions after forward propagation

### Technical Details
- Uses NumPy for matrix operations
- Implements sigmoid activation function
- Weights are initialized using normal distribution
- Supports batch processing through numpy.ndmin=2 transformation
