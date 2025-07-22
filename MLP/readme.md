# Multi-Layer Neural Network with L2 Regularization, Dropout, Batch Normalization (From Scratch)

## Overview

This project demonstrates a **Multi-Layer Perceptron (MLP)** built from scratch using **NumPy** and **Matplotlib** to perform a simple binary classification task. It integrates **L2 Regularization**, **Dropout**, **Batch Normalization**, **ReLU/Sigmoid activation**, and visualizes the **loss curve** during training.

---

## Architecture Summary

- **Input Layer**: 2 features
- **Hidden Layer 1**: 4 neurons → ReLU → Batch Normalization → Dropout
- **Hidden Layer 2**: 4 neurons → ReLU → Dropout
- **Output Layer**: 1 neuron → Sigmoid activation
- **Optimizer**: Gradient Descent

---

## Dataset (Binary Classification)

| X1 | X2 | Y (Label) |
|----|----|-----------|
| 0  | 0  | 0         |
| 1  | 0  | 1         |
| 1  | 1  | 0         |
| 0  | 1  | 1         |

---

## Components and Simple Math Formulas

### 1. **Linear Transformation**
Every layer computes:
Z = X · W + b
- `X`: Input matrix
- `W`: Weights matrix
- `b`: Bias vector
- `Z`: Pre-activation output

### 2. **Activation Functions**
- **ReLU (Rectified Linear Unit)**:
  A = max(0, Z)
- **Sigmoid** (for output layer):
  A = 1 / (1 + e^(-Z))
  This converts output into probabilities between 0 and 1.

### 3. **Batch Normalization**
Normalize activations after first linear layer:

Z_norm = (Z - mean) / sqrt(variance + ε)
A = scale * Z_norm + shift

- Stabilizes learning by reducing internal covariate shift.

### 4. **Dropout**
During training, randomly drops neurons to prevent overfitting:

A_dropout = A * mask / (1 - dropout_rate)
- `mask`: random binary mask
- Disabled neurons do not contribute to learning.

### 5. **L2 Regularization**
Adds penalty for large weights:
L2_penalty = λ * (||W1||² + ||W2||² + ||W3||²)
- Helps reduce overfitting by shrinking weights.

### 6. **Loss Function: Binary Cross Entropy + L2**
Loss = - (1/m) Σ [Y log(A) + (1-Y) log(1-A)] + L2_penalty
- Combines cross-entropy loss and L2 penalty.

### 7. **Backward Propagation (Gradients)**
Weights are updated by minimizing the loss:
W = W - learning_rate * dW
b = b - learning_rate * db
- Gradients (`dW`, `db`) are calculated layer-wise using chain rule.

## How It Works

✅ Forward Pass → Apply Linear → Batch Norm → ReLU/Sigmoid  
✅ Backward Pass → Compute Gradients with Dropout and L2  
✅ Weight Update → Gradient Descent  
✅ Output → Loss decreases across epochs, predictions improve

## Requirements

```bash
pip install numpy matplotlib


**Sample Loss vs Epoch curve**

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/fe06c184-2802-4814-8d17-5dd3c185b00a" />

