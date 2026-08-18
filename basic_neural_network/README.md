# Basic Neural Network from Scratch

## Overview

This project implements a basic artificial neural network from scratch using Python and NumPy.

The network is trained to recognize handwritten digits from the MNIST dataset.

No machine learning or deep learning framework such as TensorFlow, PyTorch, Keras, or Scikit-learn is used for the neural network implementation.

## Technologies Used

- Python
- NumPy
- Matplotlib
- MNIST Dataset

## Dataset

The MNIST dataset contains handwritten digits from 0 to 9.

- Training images: 60,000
- Test images: 10,000
- Image size: 28 × 28 pixels
- Number of classes: 10

## Neural Network Architecture

Input Layer
- 784 neurons
- 28 × 28 image flattened into 784 features

Hidden Layer
- 128 neurons
- ReLU activation

Output Layer
- 10 neurons
- Softmax activation

## Training

The network uses:

- Forward propagation
- Cross-entropy loss
- Backpropagation
- Gradient descent
- Mini-batch training

Training configuration:

- Epochs: 5
- Batch size: 64
- Learning rate: 0.1

## Preprocessing

Each MNIST image originally contains pixel values from 0 to 255.

The values are normalized to:

0 → 0.0

255 → 1.0

Each 28 × 28 image is then flattened into a vector of 784 values.

## Results

Training Accuracy: 96.90%

Test Accuracy: 96.53%

Final Loss: 0.1220

## Sample Prediction

The model successfully predicts most sample handwritten digits.

Example:

Predicted = 7
Actual = 7

Predicted = 2
Actual = 2

Predicted = 1
Actual = 1

## Project Structure

basic_neural_network/

├── neural_network.py

├── requirements.txt

├── README.md

└── data/

    ├── train-images-idx3-ubyte.gz
    ├── train-labels-idx1-ubyte.gz
    ├── t10k-images-idx3-ubyte.gz
    └── t10k-labels-idx1-ubyte.gz

## How to Run

Create a virtual environment:

```bash
python -m venv venv