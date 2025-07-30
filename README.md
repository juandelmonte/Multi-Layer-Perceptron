# Multi-Layer Perceptron (MLP) – Built from Scratch with NumPy

This project was a personal learning exercise I undertook a few years ago, back when I was just beginning to explore Python and Data Science. My goal was to deepen my understanding of how neural networks function beneath the surface—so I built a simple Multi-Layer Perceptron (MLP) entirely from scratch using NumPy. No machine learning libraries, no shortcuts—just raw matrix math and logic.

It wasn’t meant to be production-ready (and looking at how I structured it three years later, I can clearly see I wasn’t ready for that yet). Instead, the focus was on grasping core concepts like backpropagation, stochastic gradient descent (SGD), and the mechanics of how neural networks learn from data. Writing everything manually gave me a much clearer picture of how MLPs actually work.

To test the model’s flexibility, I applied it to a few classic problems: handwriting recognition, yacht hydrodynamics prediction, and fraud detection using autoencoders. These experiments were exploratory—not benchmarks—but they helped me validate the architecture and training logic.

I’m sharing this code publicly as part of my portfolio to showcase foundational machine learning principles. If you're curious about how neural networks operate at a low level, or you're learning too, I hope this project offers some insight. Feel free to explore it!

On a personal note: when I built this, ChatGPT hadn’t launched yet. I remember thinking through every component carefully. Tools like AI assistants make things easier now, but easy might not always mean better. If you're serious about understanding neural networks, I strongly recommend revisiting the fundamentals of linear algebra and mainly calculus. It’s not magic (even though it felt like that the first time my model learned something) — it’s just math.


## Installation

It is recommended to use a virtual environment to avoid conflicts with other packages.

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:

- On Windows:
```bash
venv\Scripts\activate
```

3. Install the package in editable mode:

```bash
pip install -e .
```

## Usage

After installation, you can import and use the MLP as follows:

```python
from simple_multi_layer_perceptron import MLP

# Create an MLP with specified layer sizes
# For example, input layer with 784 neurons, two hidden layers with 60 neurons each, and output layer with 10 neurons
model = MLP([784, 60, 60, 10], weights_multiplier=0.01)

# Set training data
model.setData(input_train, output_train)

# Train the model
model.learning_rate = 0.2
model.train(epochs=2000, batch_size=200, display_step=20)

# Evaluate on test data
model.setData(input_test, output_test)
cost = model.getCost()
accuracy = getAccuracy(model)
```

### Basic Workflow

1. **Initialization**: Create an MLP instance with a list of layer sizes. The first number is the input size, the last is the output size, and the middle ones are hidden layers.

2. **Data Preparation**: Use helper functions to encode your data into suitable input/output formats.

3. **Training**: Set the training data, adjust hyperparameters like learning rate, and call `train()` with number of epochs, batch size, and display frequency.

4. **Evaluation**: Switch to test/validation data and compute cost and accuracy.

### MLP Concepts Addressed

The examples demonstrate key MLP concepts:

- **Feedforward Propagation**: How inputs flow through the network layers.
- **Backpropagation**: The algorithm for computing gradients and updating weights.
- **Activation Functions**: Logistic (sigmoid) for classification, hyperbolic tangent for regression.
- **Loss Functions**: Cross-entropy for classification, mean squared error for regression.
- **Stochastic Gradient Descent (SGD)**: Batch-based weight updates during training.

## Examples

### MNIST Digits Classification

Located in `examples/mnistDigits_Classification.py`, this example trains an MLP to recognize handwritten digits from the MNIST dataset.

- **Task**: Multi-class classification (10 classes: digits 0-9).
- **Data**: 60,000 training samples, 10,000 test samples.
- **Network Architecture**: 784 input neurons (28x28 images), two hidden layers of 60 neurons each, 10 output neurons.
- **Key Features**: Uses logistic activation on the output layer and cross-entropy loss. Demonstrates training with batches and evaluating accuracy.

### Yacht Design Regression

Located in `examples/yatchDesign_Regression.py`, this example predicts yacht resistance based on hydrodynamic parameters.

- **Task**: Regression (predicting a continuous value).
- **Data**: Yacht hydrodynamics dataset with features like length, beam, etc.
- **Network Architecture**: 6 input features, one hidden layer of 4 neurons, 1 output neuron.
- **Key Features**: Uses hyperbolic tangent activation and demonstrates data normalization, train/validation split, and R-squared evaluation for regression performance.
