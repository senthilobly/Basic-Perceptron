# Simple Perceptron for AND Gate with Visualization

## Overview

This project demonstrates a basic implementation of a **Perceptron** algorithm from scratch using Python and NumPy to learn the **AND Gate** logic. The training process is visualized using Matplotlib, showing how the perceptron learns to predict correct outputs over epochs.

## Features

- Implements a **single-layer perceptron** from scratch
- Learns the **AND logic gate** without any machine learning libraries
- Step activation function for binary classification
- **Epoch-wise visualization** of true vs predicted outputs using line plots
- Lightweight code ideal for beginners to understand perceptron fundamentals

## AND Gate Truth Table

| Input X1 | Input X2 | Output (AND) |
|-----------|-----------|----------------|
| 0         | 0         | 0              |
| 0         | 1         | 0              |
| 1         | 0         | 0              |
| 1         | 1         | 1              |

## How It Works

1. **Initialize** random weights and bias.
2. **Step Activation Function** is used to output `1` if weighted sum exceeds zero, else `0`.
3. **Prediction Function** calculates the output for a given input.
4. **Training Loop** updates weights and bias based on prediction error using a simple learning rule.
5. After each epoch, it **plots the True vs Predicted outputs** for visualization.

## Code Explanation

- **step(z)**: Activation function that returns 1 if `z > 0` else 0.
- **predict(x)**: Returns the prediction for an input sample.
- **train(X, y, lr, epochs)**: Trains the perceptron using the Perceptron Learning Rule and visualizes learning progress.

## Usage

### Requirements

- Python 3.x
- NumPy
- Matplotlib

### Install Dependencies

```bash
pip install numpy matplotlib
