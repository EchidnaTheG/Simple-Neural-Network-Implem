# Simple Neural Network Implementation

A basic implementation of a single-neuron neural network that learns to perform linear regression.

## Overview

This project implements a simple neural network that:
- Uses a single neuron to learn linear relationships
- Performs gradient descent optimization
- Uses Mean Squared Error (MSE) as the loss function
- Demonstrates basic concepts of neural network training

## Components

- `main.py`: Main implementation file containing:
  - Forward propagation functions (`yfunction`, `zfunction`)
  - Loss calculation (MSE)
  - Backward propagation with gradient descent
  - Training loop
  - Prediction function

## Example Usage

```python
# Training data (x -> 3x relationship)
training_data = [0.0, 1.0, 2.0, 3.0]
expected_values = [0.0, 3.0, 6.0, 9.0]

# Train the model
training(training_data, expected_values, epochs=100)

# Make predictions
predict(4.0)  # Should output close to 12.0
